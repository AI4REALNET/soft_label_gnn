import logging
import os
import random
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union, Tuple, Optional

import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv

from gnn.obs_converter import obs_to_vect
from evaluation.utilities import split_and_execute_action, find_best_line_to_reconnect
from evaluation.general_tutor import GeneralTutor
import pandas as pd
import h5py

def save_batch_h5(save_path: Path, states, actions_list, rhos_list):
    """Save a batch of states and their corresponding actions and max_rhos in a grouped HDF5 format.

    Args:
        save_path: The path to the HDF5 file where the data should be saved/appended.
        states: A list or 2D numpy array of states, where each state is a 1D array.
        actions_list: A list of lists, where each inner list contains action IDs for a state.
        rhos_list: A list of lists, where each inner list contains max_rho values for each action in actions_list.

    Returns:
        None.
    """
    # Ensure the states are a 2D numpy array
    states = np.array(states)
    if states.ndim == 1:  # Single state provided in batch
        states = np.expand_dims(states, axis=0)

    if save_path.is_dir():
        save_path = save_path / "soft_targets.h5"

    with h5py.File(save_path, 'a') as f:
        # Determine the next group index to maintain unique group names
        starting_index = len(f.keys())

        for i, (state, act_ids, max_rhos) in enumerate(zip(states, actions_list, rhos_list)):
            group_name = f"state_{starting_index + i}"
            group = f.create_group(group_name)

            # Save state, action_ids, and max_rhos inside the group
            group.create_dataset('state', data=state, compression='gzip')
            group.create_dataset('action_ids', data=np.array(act_ids), compression='gzip')
            group.create_dataset('max_rhos', data=np.array(max_rhos), compression='gzip')

def collect_tutor_experience_one_chronic(
        action_paths: Union[Path, List[Path]],
        chronics_id: int,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        seed: Optional[int] = None,
        enable_logging: bool = True,
        subset: Optional[Union[bool, str]] = False,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {}
):
    """Collect tutor experience of one chronic.

    Args:
        action_paths: List of Paths for the tutor.
        chronics_id: Number of chronic to run.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        seed: Whether to init the Grid2Op env with a seed
        enable_logging: Whether to log the Tutor experience search.
        subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected. Further note, that it is possible to say "graph" in order to get the connectivity_matrix as well.
        TutorAgent: Tutor Agent which should be used for the search.
        tutor_kwargs: Additional arguments for the tutor, e.g. the max rho or the topology argument.
        env_kwargs: Optional arguments that should be used when initializing the environment.

    Returns:
        None.
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        env = grid2op.make(dataset=env_name_path, backend=backend,**env_kwargs)
    except ImportError:  # noqa
        env = grid2op.make(dataset=env_name_path,**env_kwargs)
        logging.warning("Not using lightsim2grid! Operation will be slow!")

    if seed:
        env.seed(seed)

    env.set_id(chronics_id)
    env.reset()
    logging.info(f"current chronic:{env.chronics_handler.get_name()}")

    # After initializing the environment, let's init the tutor
    if tutor_kwargs is None:
        tutor_kwargs = {}
    else:
        logging.info(f"Run Tutor with these additional kwargs {tutor_kwargs}")

    tutor = TutorAgent(action_space=env.action_space, action_space_file=action_paths, **tutor_kwargs)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    done, step, obs = False, 0, env.get_obs()
    if subset:
        vect_obs = obs_to_vect(obs)
    elif subset == "graph":
        vect_obs = obs_to_vect(obs, True)
    else:
        vect_obs = obs.to_vect()

    records = np.zeros((1, 1 + len(vect_obs)), dtype=np.float32)

    while not done:
        action, idx = tutor.act_with_id(obs)

        if isinstance(action,np.ndarray):
            action:grid2op.Action.BaseAction = tutor.action_space.from_vect(action)

        if action.as_dict()!={} and (idx != -1):

            # Note that we exclude the TOPO Actions of the Tutor!!!
            if subset:
                vect_obs = obs_to_vect(obs)
            elif subset == "graph":
                vect_obs = obs_to_vect(obs, True)
            else:
                vect_obs = obs.to_vect()

            records = np.concatenate(
                (records, np.hstack([idx, vect_obs]).astype(np.float32).reshape(1, -1)), axis=0
            )

            # Execute Action:
            # This method does up to three steps and returns the output
            obs, _, done, _ = split_and_execute_action(env=env, action_vect=action.to_vect())



        else:
            # Use Do-Nothing Action
            act_with_line = find_best_line_to_reconnect(obs, env.action_space({}))
            obs, _, done, _ = env.step(act_with_line)

        step = env.nb_time_step

    logging.info(f"game over at step-{step}")

    return records


def collect_tutor_experience_one_chronic_return_all(
        save_path: Union[Path, List[Path]],
        action_paths: Union[Path, List[Path]],
        chronics_id: int,
        env_name_path: Union[Path, str] = "l2rpn_neurips_2020_track1_small",
        seed: Optional[int] = None,
        enable_logging: bool = True,
        subset: Optional[Union[bool, str]] = False,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {}
):
    """Collect tutor experience of one chronic.

    Args:
        save_path: Path where the exp should be saved to
        action_paths: List of Paths for the tutor.
        chronics_id: Number of chronic to run.
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        seed: Whether to init the Grid2Op env with a seed
        enable_logging: Whether to log the Tutor experience search.
        subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected. Further note, that it is possible to say "graph" in order to get the connectivity_matrix as well.
        TutorAgent: Tutor Agent which should be used for the search.
        tutor_kwargs: Additional arguments for the tutor, e.g. the max rho or the topology argument.
        env_kwargs: Optional arguments that should be used when initializing the environment.

    Returns:
        None.
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend

        backend = LightSimBackend()
        env = grid2op.make(dataset=env_name_path, backend=backend,**env_kwargs)
    except ImportError:  # noqa
        env = grid2op.make(dataset=env_name_path,**env_kwargs)
        logging.warning("Not using lightsim2grid! Operation will be slow!")

    if seed:
        env.seed(seed)

    env.set_id(chronics_id)
    env.reset()
    logging.info(f"current chronic:{env.chronics_handler.get_name()}")

    # After initializing the environment, let's init the tutor
    if tutor_kwargs is None:
        tutor_kwargs = {}
    else:
        logging.info(f"Run Tutor with these additional kwargs {tutor_kwargs}")

    tutor = TutorAgent(action_space=env.action_space, action_space_file=action_paths, **tutor_kwargs)
    # first col for label which is the action index, remaining cols for feature (observation.to_vect())
    done, step, obs = False, 0, env.get_obs()
    if subset:
        vect_obs = obs_to_vect(obs)
    elif subset == "graph":
        vect_obs = obs_to_vect(obs, True)
    else:
        vect_obs = obs.to_vect()


        obs_ls = []
    id_ls = []
    rho_ls = []

    while not done:
        action, idx, action_rho_list = tutor.act_with_id_return_all(obs)
        
        if isinstance(action,np.ndarray):
            action:grid2op.Action.BaseAction = tutor.action_space.from_vect(action)

        if action.as_dict()!={} and (idx != -1):

            # Note that we exclude the TOPO Actions of the Tutor!!!
            if subset:
                vect_obs = obs_to_vect(obs)
            elif subset == "graph":
                vect_obs = obs_to_vect(obs, True)
            else:
                vect_obs = obs.to_vect()

            obs_ls.append(vect_obs.astype(np.float32).reshape(1, -1).tolist())

            id_array, rho_array = zip(*action_rho_list)
            id_ls.append(id_array)
            rho_ls.append(rho_array)
            # Execute Action:
            # This method does up to three steps and returns the output
            obs, _, done, _ = split_and_execute_action(env=env, action_vect=action.to_vect())

        else:
            # Use Do-Nothing Action
            act_with_line = find_best_line_to_reconnect(obs, env.action_space({}))
            obs, _, done, _ = env.step(act_with_line)

        step = env.nb_time_step
        save_batch_h5(save_path, obs_ls, id_ls, rho_ls)
        obs_ls = []
        id_ls = []
        rho_ls = []

    logging.info(f"game over at step-{step}")
    

    return [] #obs_ls, id_ls, rho_ls

def generate_tutor_experience(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        subset: Optional[Union[bool, str]] = False,
        seed: Optional[int] = None,
        TutorAgent: BaseAgent = GeneralTutor,
        tutor_kwargs: Optional[dict] = {},
        env_kwargs: Optional[dict] = {},
        return_all: Optional[bool] = False 
):
    """Method to run the Tutor in parallel.

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. If num_sample is smaller than num chronics,
            a subset is taken. If it is larger, the chronics are sampled with replacement.
        subset: Optional argument, whether the observations should be filtered when saved.
            The default version saves the observations according to obs.to_vect(), however if
            subset is set to True, then only the all observations regarding the lines, busses, generators and loads are
            selected. Further, note that it is possible to say graph in order to get the connectivity_matrix as well.
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments
        TutorAgent: Tutor Agent which should be used for the search, default is the GeneralTutor.
        tutor_kwargs: Optional arguments that should be passed to the tutor agent.
        env_kwargs: Optional arguments that should be used when initializing the environment.

    Returns:
        None, saves results as numpy file.

    """
    log_format = "(%(asctime)s) [%(name)-10s] %(levelname)8s: %(message)s [%(filename)s:%(lineno)s]"
    logging.basicConfig(level=logging.INFO, format=log_format)

    if jobs == -1:
        jobs = os.cpu_count()

    tasks = []

    # Make sure we can initialize the environment
    # This also makes sure that the environment actually exits or gets downloaded
    env: BaseEnv = grid2op.make(env_name_path)
    chronics_path = env.chronics_handler.path
    if chronics_path is None:
        raise ValueError(f"Can't determine chronics path of given environment {env_name_path}")

    if num_chronics is None:
        num_chronics = len(os.listdir(chronics_path))

    if num_sample:
        if num_sample <= num_chronics:
            sampled_chronics = random.sample(range(num_chronics), num_sample)
        else:
            sampled_chronics = random.choices(np.arange(num_chronics), k=num_sample)
    else:
        sampled_chronics = np.arange(num_chronics)

    if save_path.is_dir():
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        save_path = save_path / f"tutor_experience_return_all_{now}.h5"
    for chronic_id in sampled_chronics:
        if return_all:
            tasks.append((save_path, action_paths, chronic_id, env_name_path, seed, True, subset, TutorAgent, tutor_kwargs, env_kwargs))
        else:
            tasks.append((action_paths, chronic_id, env_name_path, seed, True, subset, TutorAgent, tutor_kwargs, env_kwargs))

    if jobs == 1:
        # This makes debugging easier since we don't fork into multiple processes
        logging.info(f"The following {len(tasks)} tasks will executed sequentially: {tasks}")
        out_result = []
        for task in tasks:
            if return_all:
                out_result.append(collect_tutor_experience_one_chronic_return_all(*task))
            else:
                out_result.append(collect_tutor_experience_one_chronic(*task))

    else:
        logging.info(f"The following {len(tasks)} tasks will be distributed to a pool of {jobs} workers:")
        start = time.time()
        with Pool(jobs) as p:
            if return_all:
                out_result = p.starmap(collect_tutor_experience_one_chronic_return_all, tasks)
            else:
                out_result = p.starmap(collect_tutor_experience_one_chronic, tasks)

        end = time.time()
        elapsed = end - start
        logging.info(f"Time: {elapsed}s")

    # Now concatenate the result:
    if return_all:
        logging.info(f"Tutor experience (return_all) has been saved to {save_path}")

        #if save_path.is_dir():
            #now = datetime.now().strftime("%d%m%Y_%H%M%S")
            #save_path = save_path / f"tutor_experience_return_all_{now}.h5"

        #for states, act_ids, max_rhos in out_result:
            #save_batch_h5(save_path, states, act_ids, max_rhos)

    #np.savez(save_path, observations=obs_ls_ls, actions_ids=id_ls_ls, max_rhos=rho_ls_ls)
    else:
        all_experience = np.concatenate(out_result, axis=0)
        if save_path.is_dir():
            now = datetime.now().strftime("%d%m%Y_%H%M%S")
            save_path = save_path / f"tutor_experience_{now}.npy"
    
        np.save(save_path, all_experience)
    logging.info(f"Tutor experience has been saved to {save_path}")



def general_tutor(
        env_name_path: Union[Path, str],
        save_path: Union[Path, str],
        action_paths: Union[Path, List[Path]],
        num_chronics: Optional[int] = None,
        num_sample: Optional[int] = None,
        jobs: int = -1,
        seed: Optional[int] = None,
        return_all: Optional[bool] = False
):
    """Method to run the general Tutor in parallel

    Args:
        env_name_path: Path to Grid2Op dataset or the standard name of it.
        save_path: Where to save the teacher_experience.
        action_paths: List of action sets (in .npy format).
        num_chronics: Total numer of chronics.
        num_sample: Length of sample from the num_chronics. With replacement!
        jobs: Number of jobs in parallel.
        seed: Whether to set a seed to the sampling of environments.

    Returns:
        None, saves results as numpy file.

    """
    generate_tutor_experience(
        env_name_path=env_name_path,
        save_path=save_path,
        action_paths=action_paths,
        num_chronics=num_chronics,
        num_sample=num_sample,
        jobs=jobs,
        seed=seed,
        return_all=return_all
    )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_path = os.getcwd()
    backend = LightSimBackend()
    env_path = "l2rpn_wcci_2022"
    env = grid2op.make(env_path, backend=backend)
    env.set_id(1)
    env.reset()
    env.chronics_handler.get_name()
    obs = env.get_obs()
    action_set = Path(example_path) / "data" /  "actions.npy"

    general_tutor(env_name_path=env_path,
                  save_path=Path(example_path) / "data" ,
                  action_paths=action_set,
                  num_chronics=None,
                  seed=23423,
                  jobs=1,
                  return_all=True)