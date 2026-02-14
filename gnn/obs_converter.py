"""This file contains features/mappings/extractions for training."""
import logging
import warnings
from typing import Optional

import grid2op
import numpy as np
import torch
from torch_geometric.utils import from_scipy_sparse_matrix


def obs_to_vect(obs: grid2op.Observation.BaseObservation, connectivity: bool = False) -> np.ndarray:
    """Method to convert only a subset of the observation to a vector.

    Args:
        obs: Original observation of Grid2Op.
        connectivity: Indicator, whether the connectivity matrix should be saved as well.

    Returns:
        Vector of the observation.

    """
    features = [
        # Timestamp Features
        [obs.month, obs.day, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week],
        # Generation and Load
        obs.gen_p,
        obs.gen_q,
        obs.gen_v,
        obs.load_p,
        obs.load_q,
        obs.load_v,
        # Raw line values
        obs.p_or,
        obs.q_or,
        obs.v_or,
        obs.a_or,
        obs.p_ex,
        obs.q_ex,
        obs.v_ex,
        obs.a_ex,
        obs.rho,
        obs.line_status,
        obs.timestep_overflow,
        # Bus information
        obs.topo_vect,
        # cool downs:
        obs.time_before_cooldown_line,
        obs.time_before_cooldown_sub,
        # maintenance
        obs.time_next_maintenance,
        obs.duration_next_maintenance,
    ]

    if connectivity:
        features.append(obs.connectivity_matrix().reshape(-1))

    return np.concatenate(features, dtype=np.float32)


def vect_to_dict(
        vect: np.ndarray, examplary_obs: grid2op.Observation.BaseObservation, connectivity: bool = False
) -> dict:
    """Method that converts the vector of the obs_to_vect method to a dictionary.
    Note that  for this one we require an observation of the environment in order to gather the correct information.

    Args:
        vect: Vector of the obs subset
        examplary_obs: One Grid2Op environment to get the correct lengths.
        connectivity: Whether to return the connectivity or not. This is only possible, if
        connectivity matrix to begin with was saved before.

    Returns:
        A dictionary of the observation.

    """
    if not isinstance(vect, np.ndarray):
        raise TypeError("vect input does not have the correct type")
    if not isinstance(examplary_obs, grid2op.Observation.BaseObservation):
        raise TypeError("examplary_obs input does not have the correct type. Please enter the observation "
                        "of the grid2op environment")

    assert len(vect.shape) < 2, "The dimensions of the vect input are not correct. Should be a vector"

    out = {
        # The first 5 are allways the same:
        "month": vect[0],
        "day": vect[1],
        "hour_of_day": vect[2],
        "minute_of_hour": vect[3],
        "day_of_week": vect[4],
    }
    i = 5
    obs_json = examplary_obs.to_json()

    for k in [
        "gen_p",
        "gen_q",
        "gen_v",
        "load_p",
        "load_q",
        "load_v",
        "p_or",
        "q_or",
        "v_or",
        "a_or",
        "p_ex",
        "q_ex",
        "v_ex",
        "a_ex",
        "rho",
        "line_status",
        "timestep_overflow",
        "topo_vect",
        "time_before_cooldown_line",
        "time_before_cooldown_sub",
        "time_next_maintenance",
        "duration_next_maintenance",
    ]:
        out[k] = vect[i: i + len(obs_json[k])]
        i += len(obs_json[k])

    if connectivity:
        if np.sqrt(len(vect[i:])) % 1 == 0:
            c_m = vect[i:].reshape(examplary_obs.connectivity_matrix().shape)
            out["connectivity_matrix"] = c_m
        else:
            logging.warning("The connectivity Matrix is not quadratic. Thus, it is not added to the dictionary")

    return out



def collect_node_and_edge_features_separated(obs):
    """
    This method collects the node and edge features from a Grid2Op observation.
    The grid is considered a graph by EACH BUS. Thus, we use the ids of each bus based on the underlying
    Grid2op environment

    In this methode, we look at each bus separately. Where each row corresponds to a bus id and the
      columns are the different values for the load/gen/lines/storage

    Args:
        obs: Observation of the current grid



    Returns: A matrix consisting of node features and the edge features. The column of the node matrix are:
    are  first the cooldown values, then the load, gen, lines_or, lines_ex and the the remaining line values.

    In comparison to collect_node_and_edge_features() we treat the p,q,v of the load/gen/lines differently.
    Further, we include storage and do not allow for specific sorting. This method is 0.001s slower.
    """
    # We extract the line/gens/... per substation. Note, we drop the first column with the substation index
    dat = [obs.get_obj_substations(substation_id=sub)[:, 1:] for sub in range(obs.n_sub)]
    # "loads","gens","lines_or","lines_ex","storage"
    global_ids = np.concatenate(dat)

    obs_dict = obs.to_dict()
    joined_data = []
    # Build information matrix:

    # 0. Cooldowns of substations and bus info
    topo_vect = obs.topo_vect.reshape(1, -1)

    cooldown_substation = np.concatenate(
        [obs_dict['cooldown']["substation"][obs.get_obj_substations(substation_id=sub)[:, 0]] for sub in
         range(obs.n_sub)]).reshape(1, -1)

    # 1. Load values: 'p', 'q', 'v'
    loads = [np.append(obs_dict["loads"][k1], 0)[global_ids[:, 0]] for k1 in obs_dict["loads"].keys()]

    # 2. Gen Values: 'p', 'q', 'v'
    gens = [np.append(obs_dict["gens"][k1], 0)[global_ids[:, 1]] for k1 in obs_dict["gens"].keys()]
    gens.append(np.append(obs_dict['gen_p_before_curtail'], 0)[global_ids[:, 1]])
    gens.append(np.append(obs_dict['curtailment'], 0)[
                    global_ids[:, 1]])  # we should not have any curtailment. However, we keep it for later in here

    # 3. Line_or values 'p', 'q', 'v',"a"
    lines_or = [np.append(obs_dict["lines_or"][k1], 0)[global_ids[:, 2]] for k1 in obs_dict["lines_or"].keys()]

    # 4. Line_ax values 'p', 'q', 'v',"a"
    lines_ex = [np.append(obs_dict["lines_ex"][k1], 0)[global_ids[:, 3]] for k1 in obs_dict["lines_ex"].keys()]

    # Add line info to all busses with either lines_or or lines_ex:
    line_mask = global_ids[:, 3].copy()
    line_mask[global_ids[:, 2] != -1] = global_ids[:, 2][global_ids[:, 2] != -1]
    line_infos = [np.append(obs_dict[k2], 0)[line_mask] for k2 in ["timestep_overflow", "line_status", "rho"]]

    # Add maintance:
    line_infos.append(np.append(obs_dict["maintenance"]['time_next_maintenance'], 0)[line_mask])
    line_infos.append(np.append(obs_dict["maintenance"]['duration_next_maintenance'], 0)[line_mask])
    line_infos.append(np.append(obs_dict['cooldown']['line'], 0)[line_mask])

    # storage
    storage = [
        np.append(obs_dict['storage_charge'], 0)[global_ids[:, 4]],
        np.append(obs_dict['storage_power_target'], 0)[global_ids[:, 4]],
        np.append(obs_dict['storage_power'], 0)[global_ids[:, 4]]
    ]

    # Join all data:
    out = np.concatenate(
        [topo_vect, cooldown_substation, loads, gens, lines_or, lines_ex, line_infos, storage]).transpose()

    out = torch.tensor(out).type(torch.FloatTensor)

    sparse_conn_mat = obs.connectivity_matrix(as_csr_matrix=True)
    edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

    return out, edge_index
