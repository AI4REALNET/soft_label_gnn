import logging
import pickle
from pathlib import Path
from typing import Union, List, Tuple, Optional
import os
import grid2op
import numpy as np
import torch
from lightsim2grid import LightSimBackend
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected

from gnn.obs_converter import collect_node_and_edge_features_separated
import sys
# from Date2Vec.Model import Date2VecConvert  # Not needed for current usage
 

#@todo clean signature of function, expacially env
def remove_zero_rows_from_experience(s:np.ndarray, a:np.ndarray, filter_out_by_obs: bool=True, env: str = "l2rpn_case14_sandbox" ):

    non_zero_idx = np.where(s[:,0]!= 0)[0]
    print(f"{s.shape[0] - non_zero_idx.shape[0]} of {len(s)} rows will be removed due to zero observation")
    s = s[non_zero_idx]
    a = a[non_zero_idx]

    if filter_out_by_obs:
        non_zero_obs_idx = []
        env = grid2op.make(env, backend=LightSimBackend())
        for i,x in enumerate(s):
            obs = env.observation_space.from_vect(x)
            if obs.line_status.any():
                non_zero_obs_idx.append(i)
        print(f"{s.shape[0] - len(non_zero_obs_idx)} of {s.shape[0]} rows will be further be removed due to game over obs")
        s = s[non_zero_obs_idx]
        a = a[non_zero_obs_idx]

    return s,a

    
class Grid2opGraphDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend())
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)

        Xy = np.concatenate([data["dn"], data["senior"], data["topo"]], axis=0)
        s_dat, a_dat = Xy[:, :-1], Xy[:, -1]
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env.name)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

        data = Data(x=node_features, edge_index=to_undirected(edge_index), y=torch.tensor(self.a[idx]).type(torch.long))

        return data

class Grid2opGraphDatasetProcessed(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        # For Grid2opGraphDatasetProcessed, we don't need the env object since we load pre-processed data
        # This avoids pickling issues with multiprocessing DataLoader
        if env is None:
            self.env = None
        elif type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend())
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.data = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """
        path = self.dataset_path / self.dataset_name
        data = torch.load(path, weights_only=False)

        if self.scaler:
            data = self.scaler.transform(data)

        return data
        
    def process(self):
        pass

    def len(self):
        return len(self.data)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        return self.data[idx]

