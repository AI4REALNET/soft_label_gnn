import logging
import os
from pathlib import Path
from typing import Optional, Union, List

import grid2op
import numpy as np
import torch
from grid2op.Agent import BaseAgent
from sklearn.base import BaseEstimator
import pickle
from gnn.obs_converter import collect_node_and_edge_features_separated
from torch_geometric.data import Data, Batch
from copy import deepcopy


from gnn.obs_converter import obs_to_vect
from evaluation.utilities import (
    find_best_line_to_reconnect,
    is_legal,
    split_action_and_return,
    simulate_action, revert_topo,
)
from gnn.gnn_models import GAT

class GNNAgent(BaseAgent):
    def __init__(
            self,
            action_space,
            model_path: Union[Path, str],
            action_space_path: Optional[Union[Path, List[Path]]] = None,
            this_directory_path: Optional[str] = "./",
            subset: Optional[bool] = False,
            scaler: Optional[BaseEstimator] = None,
            best_action_threshold: float = 0.95,
            topo: Optional[bool] = False,
            check_overload: Optional[bool] = False,
            max_action_sim: Optional[int] = 50,
            action_space_file: Optional[str] = None,
            run_with_tf: bool = True,
    ):
        """The new advanced agent.

        In contrast to the original agent, this agent enables the implementation of tuple and triple actions,
        as well as use either a keras model.

        Next to the difference in the models and actions, this agent also has the ability to transform the
        observations based on a provided scaler and/or filter them accordingly.

        Note:
            If you just want to pass this agent as submission without the CurriculumAgent, copy the content
            of the common dir into this directory. Further, add the model and actions to complete it.

        Args:
            action_space: Action Space of the Grid2Op Enviornment
            model_path: Path where to find the rllib model or Keras model
            action_space_path: path, where to find the action sets. This is required to run the agent
            this_directory_path: Path of the submission directory
            subset: Boolean, whether to filter the observation
            scaler: Optional Scaler for the neural network
            best_action_threshold: Threshold, when to stop searching for the results.
            topo: Booling indicator, whether the agent should revert to original topology if it is possible
            check_overload: Boolean, whether to simulate a stress of the generation and load
            max_action_sim: Define, how many of the actions you want to evaluate before selecting a suitable
            candidate. If you want to select all, it has to be the number of actions. For a more rapid simulation, you
            can just select fewer values.
            action_space_file: Optional alternative to action_space_path, if you want to provide the file itself.
            run_with_tf: Flag to determine if the agent should use TensorFlow (True) or PyTorch (False).
        """
        # Initialize a new agent.
        BaseAgent.__init__(self, action_space=action_space)

        if run_with_tf:
            import tensorflow as tf
            from tensorflow.keras.models import Model

        # Collect action set:
        # If action_space_file is available, we overwrite it here
        if isinstance(action_space_file,(str,Path)):
            action_space_path = Path(action_space_file)

        self.actions = self.__collect_action(
            this_directory_path=this_directory_path, action_space_path=action_space_path
        )

        self.subset = subset
        self.check_overload = check_overload
        self.run_with_tf = run_with_tf
        # Initialize model based on the framework
        if self.run_with_tf:
            # TensorFlow model loading
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile()
                logging.info(f"Successfully loaded TensorFlow model from {model_path}.")
            except (IndexError, AttributeError) as e:
                raise AttributeError(f"You are trying to load an old TensorFlow model that is not supported: {e}")
            except Exception as e:
                raise ValueError(f"Error loading TensorFlow model: {e}")
        else:
            # PyTorch model loading
            try:
                train_config_path = model_path / "train_config.pkl"
                with open(train_config_path, 'rb') as file:
                    train_conf = pickle.load(file)


                self.model = GAT(train_conf[0])
                
                # Load the saved state_dict
                state_dict = torch.load(model_path / "model.pt" , map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()

                logging.info(f"Successfully loaded PyTorch model from {model_path}.")
            except Exception as e:
                raise ValueError(f"Error loading PyTorch model: {e}")

        self.scaler = scaler
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None
        self.best_action_threshold = best_action_threshold
        self.max_action_sim = max_action_sim
        if topo:
            self.topo = topo
        else:
            self.topo = False

    def act_with_id(
            self, observation: grid2op.Observation.BaseObservation) -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Note: this method was primarily written to plug it into the generate experience methode

        Args:
            observation: Grid2Op Observation

        Returns: A suitable Grid2Op action

        """

        # Similar to the Tutor, we check whether there is some remaining action, based on previous
        # selected tuples
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure with low threshold
        if observation.rho.max() < self.best_action_threshold:  # fixed threshold
            if self.topo:
                action_array = revert_topo(self.action_space, observation,
                                           rho_limit=0.8)
                default_action = self.action_space.from_vect(action_array)
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)
            else:
                default_action = self.action_space({})
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)

            return default_action, -1

        # Now, case dangerous:
        min_rho = observation.rho.max()

        logging.info(
            f"{observation.get_time_stamp()}s, heavy load,"
            f" line-{observation.rho.argmax()}d load is {observation.rho.max()}"
        )

        idx_chosen = None

        sorted_actions = self.__get_actions(obs=observation)[:self.max_action_sim]

        for k, idx in enumerate(sorted_actions):
            action_vect = self.actions[idx, :]
            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                                    action_vect=action_vect, check_overload=self.check_overload
                                                    )
            if not valid_action:
                continue

            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                logging.info(f"take action {idx}, max-rho to {rho_max}," f" simulation times: {k + 1}")
                idx_chosen = idx
                break

            if rho_max < min_rho:
                # If we have a decrease in rho, we already save the candidate.
                min_rho = rho_max
                idx_chosen = idx

        if idx_chosen:
            self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)

        else:
            next_action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))
            idx_chosen = -1

        return next_action,idx_chosen

    def act( self, observation: grid2op.Observation.BaseObservation, reward: float, done: bool)\
            -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Args:
            observation: Grid2Op Observation
            reward: Reward of the previous action
            done: Whether the agent is done

        Returns: A suitable Grid2Op action

        """
        # We do not need done or the reward!

        action, _ = self.act_with_id(observation)
        return action

    def reset(self, obs: grid2op.Observation.BaseObservation):
        """ Resetting the agent.

        Args:
            obs:

        Returns:

        """
        self.next_actions = None

    def __collect_action(self, this_directory_path: str, action_space_path: Union[Path, List[Path]]) -> np.ndarray:
        """Check the action space path for the different action set.

        Args:
            this_directory_path: Directory of the submission files
            action_space_path: Optional action space path

        Returns:

        """
        actions = None
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logging.info(f"action_space_path {action_space_path} is a file and will be loaded.")
                actions = np.load(str(action_space_path))
            elif action_space_path.is_dir():
                logging.info(
                    f"action_space_path {action_space_path} is a path. All available action files "
                    f" will be loaded."
                )
                all_action_files = [
                    act for act in os.listdir(action_space_path) if "actions" in act and ".npy" in act
                ]

                if not all_action_files:
                    raise FileNotFoundError("No actions files were found!")

                loaded_files = []
                for act in all_action_files:
                    if "actions" in act and ".npy" in act:
                        loaded_files.append(np.load(action_space_path / act))

                actions = np.concatenate(loaded_files, axis=0)

        elif isinstance(action_space_path, list):
            logging.info(f"action_space_path {action_space_path} is a list containing multiple actions.")
            for act_path in action_space_path:
                if isinstance(act_path, Path):
                    assert act_path.is_file()
                else:
                    os.path.isfile(act_path)
            loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
            actions = np.concatenate(loaded_files, axis=0)
        else:
            raise ValueError(
                f"The action_space_path variable {action_space_path} does neither consist of a single "
                f"action nor of a path where actions can be found."
            )

        return actions

    def __get_actions(self, obs: grid2op.Observation.BaseObservation):
        """ This method conducts depending on the underlying model the action method

        Args:
            obs: Input of the Grid2op Environment

        Returns: action

        """
        if self.run_with_tf:
            if isinstance(self.model, Model):
                # Newer Junior or Senior model
                sorted_actions = self.__get_keras_actions_model(obs=obs)

            else:
                # Older Model from Ray<2.4
                sorted_actions = self.__get_tf_actions(obs=obs)
        else:
            sorted_actions = self.__get_torch_actions_model(obs=obs)

        return sorted_actions

    def __get_keras_actions_model(self, obs: grid2op.Observation.BaseObservation):
        """Method to get the keras actions:

        Args:
            obs: Current observations

        Returns: numpy with the sorted actions:

        """
        # Select subset if wanted
        if isinstance(self.subset, list):
            model_input = obs.to_vect()[self.subset]
        elif self.subset:
            model_input = obs_to_vect(obs, False)
        else:
            model_input = obs.to_vect()

        if self.scaler:
            model_input = self.scaler.transform(model_input.reshape(1, -1))

        model_input = model_input.reshape((1, -1))

        if isinstance(self.model, tf.keras.models.Sequential):
            # Junior Model: Sequential model
            action_probability = self.model.predict(model_input, verbose=0)
        else:
            # Senior Model: tensorflow functional model:
            action_probability_pre_softmax, _ = self.model.predict(model_input, verbose=0)
            action_probability = tf.nn.softmax(action_probability_pre_softmax).numpy().reshape(-1)
        sorted_actions = action_probability.argsort()[::-1]
        return sorted_actions.reshape(-1)


    def __get_tf_actions(self, obs: grid2op.Observation.BaseObservation):
            """Method to get the tf actions:

            Args:
                obs: Current observations

            Returns: sorted numpy array with actions

            """
            # Select subset if wanted
            if isinstance(self.subset, list):
                model_input = obs.to_vect()[self.subset]
            elif self.subset:
                model_input = obs_to_vect(obs, False)
            else:
                model_input = obs.to_vect()

            if self.scaler:
                model_input = self.scaler.transform(model_input.reshape(1, -1)).reshape(
                    -1,
                )

            f = self.model.signatures["serving_default"]
            out = f(
                observations=tf.convert_to_tensor(model_input.reshape(1, -1)),
                timestep=tf.convert_to_tensor(0, dtype=tf.int64),
                is_training=tf.convert_to_tensor(False),
            )

            # Collect the softmax over all actions
            try:
                prob_of_action = (
                    tf.nn.softmax(out["action_dist_inputs"])
                    .numpy()
                    .reshape(
                        -1,
                    )
                )
            except AttributeError:
                poa = tf.nn.softmax(out["action_dist_inputs"])
                prob_of_action = poa.eval(session=tf.compat.v1.Session()).reshape(-1, )

            sorted_actions = prob_of_action.argsort()[::-1]
            return sorted_actions


    def __get_torch_actions_model(self, obs: grid2op.Observation.BaseObservation):
            """Method to get the torch actions (for PyTorch model).

            Args:
                obs: Current observations

            Returns: numpy array with the sorted actions

            """
            features, edge_index= collect_node_and_edge_features_separated(obs)

            if self.scaler:
                features = torch.tensor(self.scaler.transform(features.numpy()), dtype=torch.float)

            # Assuming you have tensors `x` and `edge_index`
            model_input = Data(x=features, edge_index=edge_index)

            with torch.no_grad():
                output = self.model(model_input)
                action_probabilities = torch.exp(output).numpy().reshape(-1) #need to be checked
                #action_probabilities = torch.softmax(torch.exp(output), dim=1).numpy().reshape(-1) #need to be checked

            sorted_actions = action_probabilities.argsort()[::-1]
            return sorted_actions

    def __generate_config(self, checkpoint):
        """
        Generate the configuration dictionary from a checkpoint dict.

        Args:
            checkpoint (dict): The checkpoint dictionary containing 'state_dict' and 'model_config'.

        Returns:
            dict: A dictionary containing the 'custom_config' with activation, initializer, and layer sizes.
        """
        # Initialize custom_config dictionary
        custom_config = {}

        # Extract 'model_config' from the checkpoint
        model_config = checkpoint.get('model_config', None)

        # Check if 'model_config' is not empty and extract values
        if model_config:
            activation = model_config.get('activation', 'relu')
            initializer = model_config.get('initializer', 'O')
            layer1 = model_config.get('layer1', 1000)
            layer2 = model_config.get('layer2', 1000)
            layer3 = model_config.get('layer3', 1000)
            layer4 = model_config.get('layer4', 1000)
        else:
            # Use default values if 'model_config' is empty
            activation = 'relu'
            initializer = 'O'
            layer1 = 1000
            layer2 = 1000
            layer3 = 1000
            layer4 = 1000

        custom_config['activation'] = activation
        custom_config['initializer'] = initializer
        custom_config['layer1'] = layer1
        custom_config['layer2'] = layer2
        custom_config['layer3'] = layer3
        custom_config['layer4'] = layer4

        # Extract the 'state_dict' from the checkpoint
        state_dict = checkpoint.get('state_dict', {})
        if not state_dict:
            raise ValueError("The checkpoint does not contain a 'state_dict'.")

        return {'custom_config': custom_config}


class GNNAgentN1(BaseAgent):
    def __init__(
            self,
            action_space,
            model_path: Union[Path, str],
            action_space_path: Optional[Union[Path, List[Path]]] = None,
            this_directory_path: Optional[str] = "./",
            subset: Optional[bool] = False,
            scaler: Optional[BaseEstimator] = None,
            best_action_threshold: float = 0.95,
            topo: Optional[bool] = False,
            check_overload: Optional[bool] = False,
            max_action_sim: Optional[int] = 50,
            top_k: Optional[int] = 10,
            action_space_file: Optional[str] = None,
            run_with_tf: bool = True,
    ):
        """The new advanced agent.

        In contrast to the original agent, this agent enables the implementation of tuple and triple actions,
        as well as use either a keras model.

        Next to the difference in the models and actions, this agent also has the ability to transform the
        observations based on a provided scaler and/or filter them accordingly.

        Note:
            If you just want to pass this agent as submission without the CurriculumAgent, copy the content
            of the common dir into this directory. Further, add the model and actions to complete it.

        Args:
            action_space: Action Space of the Grid2Op Enviornment
            model_path: Path where to find the rllib model or Keras model
            action_space_path: path, where to find the action sets. This is required to run the agent
            this_directory_path: Path of the submission directory
            subset: Boolean, whether to filter the observation
            scaler: Optional Scaler for the neural network
            best_action_threshold: Threshold, when to stop searching for the results.
            topo: Booling indicator, whether the agent should revert to original topology if it is possible
            check_overload: Boolean, whether to simulate a stress of the generation and load
            max_action_sim: Define, how many of the actions you want to evaluate before selecting a suitable
            candidate. If you want to select all, it has to be the number of actions. For a more rapid simulation, you
            can just select fewer values.
            action_space_file: Optional alternative to action_space_path, if you want to provide the file itself.
            run_with_tf: Flag to determine if the agent should use TensorFlow (True) or PyTorch (False).
        """
        # Initialize a new agent.
        BaseAgent.__init__(self, action_space=action_space)

        # Collect action set:
        # If action_space_file is available, we overwrite it here
        if isinstance(action_space_file,(str,Path)):
            action_space_path = Path(action_space_file)

        self.actions = self.__collect_action(
            this_directory_path=this_directory_path, action_space_path=action_space_path
        )

        self.subset = subset
        self.check_overload = check_overload
        self.run_with_tf = run_with_tf
        # Initialize model based on the framework
        if self.run_with_tf:
            # TensorFlow model loading
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile()
                logging.info(f"Successfully loaded TensorFlow model from {model_path}.")
            except (IndexError, AttributeError) as e:
                raise AttributeError(f"You are trying to load an old TensorFlow model that is not supported: {e}")
            except Exception as e:
                raise ValueError(f"Error loading TensorFlow model: {e}")
        else:
            # PyTorch model loading
            try:
                train_config_path = model_path / "train_config.pkl"
                with open(train_config_path, 'rb') as file:
                    train_conf = pickle.load(file)

                self.model = GAT(train_conf[0])
                
                # Load the saved state_dict
                state_dict = torch.load(model_path / "model.pt" , map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()

                logging.info(f"Successfully loaded PyTorch model from {model_path}.")
            except Exception as e:
                raise ValueError(f"Error loading PyTorch model: {e}")


        self.scaler = scaler
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None
        self.best_action_threshold = best_action_threshold
        self.max_action_sim = max_action_sim

        self.top_k = top_k
        self.lines_to_attack = [106,88,162,68,108,180,160,136,141,131,121,125,126,110,154,81,175,43,33,37,62,61]#93

        if topo:
            self.topo = topo
        else:
            self.topo = False

    def calculate_attacked_max_rho(self, obs, action) -> float:
        """Method to run through all lines and check the maximum rho if the lines is disconnected.
        Note that if the game is done, the rho will be set to np.inf
        This is to measure how robust the supplied action is against the attacks, the lower, the better.
    
        Args:
            obs: Observation of the Grid2Op Env.
            action: Current selected action.
    
        Returns:
            The maximum rho of the action.
    
        """
        action = self.action_space.from_vect(action)

        p_max_max = 0.0
        for line in self.lines_to_attack:
            new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
            new_line_status_array[line] = -1
            disconnect_action = deepcopy(action)
            disconnect_action.update({"set_line_status": new_line_status_array})
            obs_new, _, done, _ = obs.simulate(disconnect_action)
            rho_max = obs_new.rho.max()
            if rho_max > p_max_max and done is False:
                p_max_max = rho_max
            if done:
                p_max_max = np.inf
    
        return p_max_max

    
    def act_with_id(
        self, observation: grid2op.Observation.BaseObservation
    ) -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Note: this method was primarily written to plug it into the generate experience methode

        Args:
            observation: Grid2Op Observation

        Returns: A suitable Grid2Op action

        """

        # Similar to the Tutor, we check whether there is some remaining action, based on previous
        # selected tuples
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure with low threshold
        if observation.rho.max() < self.best_action_threshold:  # fixed threshold
            if self.topo:
                action_array = revert_topo(self.action_space, observation,
                                           rho_limit=0.8)
                default_action = self.action_space.from_vect(action_array)
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)
            else:
                default_action = self.action_space({})
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)

            return default_action, -1

        # Now, case dangerous:
        min_rho = observation.rho.max()

        logging.info(
            f"{observation.get_time_stamp()}s, heavy load,"
            f" line-{observation.rho.argmax()}d load is {observation.rho.max()}"
        )

        idx_chosen = None

        sorted_actions = self.__get_actions(obs=observation)[:self.max_action_sim]

        n1_actions = []
        n1_maxrho = 100

        for k, idx in enumerate(sorted_actions[0:self.top_k]):
            action_vect = self.actions[idx, :]

            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                        action_vect=action_vect, check_overload=self.check_overload
                                        )
            if not valid_action:
                continue

            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                n1_actions.append(idx)
        
        if n1_actions:
            logging.info(f"found multiple suitable actions for n-1 [{n1_actions}], running n-1 search")

            for idx in n1_actions:
                act = self.actions[idx, :]
                n1_rho = self.calculate_attacked_max_rho(observation,act)
                if n1_rho < n1_maxrho:
                    # If we have a decrease in rho, we already save the candidate.
                    n1_maxrho = n1_rho
                    idx_chosen = idx
            
            if idx_chosen:
                logging.info(f"found suitable action {idx_chosen} with n-1 with n-1 max_rho {n1_maxrho}")

                self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                return next_action,idx_chosen



        
        for k, idx in enumerate(sorted_actions):
            action_vect = self.actions[idx, :]
            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                                    action_vect=action_vect, check_overload=self.check_overload
                                                    )
            if not valid_action:
                continue

            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                logging.info(f"take action {idx}, max-rho to {rho_max}," f" simulation times: {k + 1}")
                idx_chosen = idx
                break

            if rho_max < min_rho:
                # If we have a decrease in rho, we already save the candidate.
                min_rho = rho_max
                idx_chosen = idx

        if idx_chosen:
            self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)

        else:
            next_action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))
            idx_chosen = -1

        return next_action,idx_chosen

    def act( self, observation: grid2op.Observation.BaseObservation, reward: float, done: bool)\
            -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Args:
            observation: Grid2Op Observation
            reward: Reward of the previous action
            done: Whether the agent is done

        Returns: A suitable Grid2Op action

        """
        # We do not need done or the reward!

        action, _ = self.act_with_id(observation)
        return action

    def reset(self, obs: grid2op.Observation.BaseObservation):
        """ Resetting the agent.

        Args:
            obs:

        Returns:

        """
        self.next_actions = None

    def __collect_action(self, this_directory_path: str, action_space_path: Union[Path, List[Path]]) -> np.ndarray:
        """Check the action space path for the different action set.

        Args:
            this_directory_path: Directory of the submission files
            action_space_path: Optional action space path

        Returns:

        """
        actions = None
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logging.info(f"action_space_path {action_space_path} is a file and will be loaded.")
                actions = np.load(str(action_space_path))
            elif action_space_path.is_dir():
                logging.info(
                    f"action_space_path {action_space_path} is a path. All available action files "
                    f" will be loaded."
                )
                all_action_files = [
                    act for act in os.listdir(action_space_path) if "actions" in act and ".npy" in act
                ]

                if not all_action_files:
                    raise FileNotFoundError("No actions files were found!")

                loaded_files = []
                for act in all_action_files:
                    if "actions" in act and ".npy" in act:
                        loaded_files.append(np.load(action_space_path / act))

                actions = np.concatenate(loaded_files, axis=0)

        elif isinstance(action_space_path, list):
            logging.info(f"action_space_path {action_space_path} is a list containing multiple actions.")
            for act_path in action_space_path:
                if isinstance(act_path, Path):
                    assert act_path.is_file()
                else:
                    os.path.isfile(act_path)
            loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
            actions = np.concatenate(loaded_files, axis=0)
        else:
            raise ValueError(
                f"The action_space_path variable {action_space_path} does neither consist of a single "
                f"action nor of a path where actions can be found."
            )

        return actions

    def __get_actions(self, obs: grid2op.Observation.BaseObservation):
        """ This method conducts depending on the underlying model the action method

        Args:
            obs: Input of the Grid2op Environment

        Returns: action

        """
        if self.run_with_tf:
            if isinstance(self.model, Model):
                # Newer Junior or Senior model
                sorted_actions = self.__get_keras_actions_model(obs=obs)

            else:
                # Older Model from Ray<2.4
                sorted_actions = self.__get_tf_actions(obs=obs)
        else:
            sorted_actions = self.__get_torch_actions_model(obs=obs)

        return sorted_actions

    def __get_keras_actions_model(self, obs: grid2op.Observation.BaseObservation):
        """Method to get the keras actions:

        Args:
            obs: Current observations

        Returns: numpy with the sorted actions:

        """
        # Select subset if wanted
        if isinstance(self.subset, list):
            model_input = obs.to_vect()[self.subset]
        elif self.subset:
            model_input = obs_to_vect(obs, False)
        else:
            model_input = obs.to_vect()

        if self.scaler:
            model_input = self.scaler.transform(model_input.reshape(1, -1))

        model_input = model_input.reshape((1, -1))

        if isinstance(self.model, tf.keras.models.Sequential):
            # Junior Model: Sequential model
            action_probability = self.model.predict(model_input, verbose=0)
        else:
            # Senior Model: tensorflow functional model:
            action_probability_pre_softmax, _ = self.model.predict(model_input, verbose=0)
            action_probability = tf.nn.softmax(action_probability_pre_softmax).numpy().reshape(-1)
        sorted_actions = action_probability.argsort()[::-1]
        return sorted_actions.reshape(-1)


    def __get_tf_actions(self, obs: grid2op.Observation.BaseObservation):
            """Method to get the tf actions:

            Args:
                obs: Current observations

            Returns: sorted numpy array with actions

            """
            # Select subset if wanted
            if isinstance(self.subset, list):
                model_input = obs.to_vect()[self.subset]
            elif self.subset:
                model_input = obs_to_vect(obs, False)
            else:
                model_input = obs.to_vect()

            if self.scaler:
                model_input = self.scaler.transform(model_input.reshape(1, -1)).reshape(
                    -1,
                )

            f = self.model.signatures["serving_default"]
            out = f(
                observations=tf.convert_to_tensor(model_input.reshape(1, -1)),
                timestep=tf.convert_to_tensor(0, dtype=tf.int64),
                is_training=tf.convert_to_tensor(False),
            )

            # Collect the softmax over all actions
            try:
                prob_of_action = (
                    tf.nn.softmax(out["action_dist_inputs"])
                    .numpy()
                    .reshape(
                        -1,
                    )
                )
            except AttributeError:
                poa = tf.nn.softmax(out["action_dist_inputs"])
                prob_of_action = poa.eval(session=tf.compat.v1.Session()).reshape(-1, )

            sorted_actions = prob_of_action.argsort()[::-1]
            return sorted_actions


    def __get_torch_actions_model(self, obs: grid2op.Observation.BaseObservation):
            """Method to get the torch actions (for PyTorch model).

            Args:
                obs: Current observations

            Returns: numpy array with the sorted actions

            """
            features, edge_index= collect_node_and_edge_features_separated(obs)

            if self.scaler:
                features = torch.tensor(self.scaler.transform(features.numpy()), dtype=torch.float)

            # Assuming you have tensors `x` and `edge_index`
            model_input = Data(x=features, edge_index=edge_index)

            with torch.no_grad():
                output = self.model(model_input)
                action_probabilities = torch.exp(output).numpy().reshape(-1) #need to be checked
                #action_probabilities = torch.softmax(torch.exp(output), dim=1).numpy().reshape(-1) #need to be checked

            sorted_actions = action_probabilities.argsort()[::-1]
            return sorted_actions

    def __generate_config(self, checkpoint):
        """
        Generate the configuration dictionary from a checkpoint dict.

        Args:
            checkpoint (dict): The checkpoint dictionary containing 'state_dict' and 'model_config'.

        Returns:
            dict: A dictionary containing the 'custom_config' with activation, initializer, and layer sizes.
        """
        # Initialize custom_config dictionary
        custom_config = {}

        # Extract 'model_config' from the checkpoint
        model_config = checkpoint.get('model_config', None)

        # Check if 'model_config' is not empty and extract values
        if model_config:
            activation = model_config.get('activation', 'relu')
            initializer = model_config.get('initializer', 'O')
            layer1 = model_config.get('layer1', 1000)
            layer2 = model_config.get('layer2', 1000)
            layer3 = model_config.get('layer3', 1000)
            layer4 = model_config.get('layer4', 1000)
        else:
            # Use default values if 'model_config' is empty
            activation = 'relu'
            initializer = 'O'
            layer1 = 1000
            layer2 = 1000
            layer3 = 1000
            layer4 = 1000

        custom_config['activation'] = activation
        custom_config['initializer'] = initializer
        custom_config['layer1'] = layer1
        custom_config['layer2'] = layer2
        custom_config['layer3'] = layer3
        custom_config['layer4'] = layer4

        # Extract the 'state_dict' from the checkpoint
        state_dict = checkpoint.get('state_dict', {})
        if not state_dict:
            raise ValueError("The checkpoint does not contain a 'state_dict'.")

        return {'custom_config': custom_config}