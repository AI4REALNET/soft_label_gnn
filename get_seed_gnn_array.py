"""
This file is running through various seeds, to check the configurations fo different seed s
"""
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import defopt
import grid2op
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend

from evaluation.score_agent import load_or_run, render_report
from evaluation.GNNAgent import GNNAgent, GNNAgentN1

DATA_PATH = "./" # Adjust these paths to get to correct data path of your env and where you want to save stuff
 # Where can we find the 2022 dir and the models
HOME_PATH = "./eval"


def run_evaluation_of_env(agent_dict: dict, env: BaseEnv, out_path: Path, seed: Optional[int], nb_process) -> Tuple[
    dict, dict]:
    """ Calculate the performance of the provided agents (from the agents dicts)

    Note: The do nothing agent is always provided!

    Args:
        agent_dict: Dictionary containing multiple agents
        env: Grid2Op environment
        out_path: path, where to save the results and cached results
        seed: Optional seed for execution.

    Returns: None

    """
    if out_path.is_dir() is False:
        os.mkdir(out_path)

    number_of_runs = len(os.listdir(env.chronics_handler.path))

    do_nothing_agent = DoNothingAgent(env.action_space)
    
    dn_report = load_or_run(agent=do_nothing_agent, env=env, output_path=out_path, name="DoNothing",
                            number_episodes=number_of_runs, seed=seed, reinit=True)
    print(f"The Do-Nothing agent has the scores of: {dn_report.score_data['all_scores']}")

    agent_res = []

    for name, agent in agent_dict.items():
        print(f"Run with Agent {name}")
        agent_res.append(load_or_run(agent, env=env, output_path=out_path, name=name,
                                     nb_processes=nb_process,
                                     number_episodes=number_of_runs, seed=seed,
                                     score_l2rpn2020=False))

        sys.stdout.flush()

    render_report(out_path / 'report.md', dn_report, agent_res)

    # collect the mean of the overall scores: 
    res_dict = {dn_report.agent_name: dn_report.avg_score}
    for agent in agent_res:
        res_dict[agent.agent_name] = agent.avg_score

    # Collect the surviving time
    surv_time = {dn_report.agent_name: dn_report.score_data['ts_survived']}
    for agent in agent_res:
        surv_time[agent.agent_name] = agent.score_data['ts_survived']

    return res_dict, surv_time


def create_agents_and_env(seed=None):
    """ Simple Method to initialize the agents in order to make Pooling work

    Returns: dictionary of agent

    """
    # Paths (delete later for privacy reasons)
    env_path = Path("/mnt/home/mhassouna/share") / "envs3/"
    
    ppath = Path(HOME_PATH)
    actions_list = ppath /  "actions.npy"
    #topo_path = Path(DATA_PATH) / "junior"/"wcci2022_topo/"
    
    
    ##############
    # Environment
    ##############
    # Note: In order for this to work, you have to duplicate your validation environment 
    # by the number of seeds you want to run. This needs to be done to ensure that the 
    # the DoNothing Stastistics are independend from each other 
    backend = LightSimBackend()
    env = grid2op.make(
        env_path  / f"l2rpn_2022_val_{seed}",
        backend=LightSimBackend())
    env.generate_classes()
    env = grid2op.make(
        env_path  / f"l2rpn_2022_val_{seed}",
        backend=LightSimBackend(), experimental_read_from_local_dir=True)


    ##############
    # Scaler
    ##############    
    
    # This scaler is made for subset=True Agents
    with open(ppath / 'scaler_all.pkl', "rb") as fp:  
        scaler_old = pickle.load(fp)
    
    
    ##############
    # Agents 
    ##############
    agent_kwargs = {"model_path": ppath / "model_soft",
                    "subset": True,
                    "this_directory_path": ppath / "res",
                    "action_space_path": actions_list,
                    "scaler": scaler_old,
                    "topo": True,
                    "max_action_sim":2030,
                    }

    gnn90 = GNNAgent(
        action_space=env.action_space,
        action_space_file=actions_list,
        best_action_threshold=0.90,
        run_with_tf = False,
        **agent_kwargs)
    
    # Adding topology Agent:
    """topo_kwargs = agent_kwargs.copy()
    topo_kwargs["topology_actions"] = topo_path / "topologies_only.npy"

    topo_agent2 = TopologyAgent2(action_space=env.action_space,
                               action_space_file=ppath / "2022" / "actions" / "actions.npy",
                               best_action_threshold=0.95,
                               topo_threshold = 0.85,
                               **topo_kwargs)"""
    
    agents = {"SoftGNN_90": gnn90
             }

    return agents, env


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    date_strftime_format = "%Y-%m-%y %H:%M:%S"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt=date_strftime_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # The seeds from defop main are from: 
    np.random.seed(8888)
    seeds = list(np.random.randint(0, 10000, 20))
    seed = seeds[int(sys.argv[1])]
    collect_scores = {}
    collect_survival_time = {}

    print(f"-------------------------- Run main with {seed} -------------------------")
    agents, env = create_agents_and_env(seed)

    print(f"Let's fuck shit up. I choose you, seed {seed}")
    res, surv_time = run_evaluation_of_env(agent_dict=agents,
                                           env=env,
                                           out_path=Path("seeds_gnn_2") / str(seed),
                                           seed=seed,
                                           nb_process=1)
    print(f"-------------------------- Done with {seed} -------------------------")
    collect_scores[seed] = res
    collect_survival_time[seed] = surv_time

    with open(f'./seeds_gnn_2/seed_res_gnn_{seed}.pkl', 'wb') as handle:
        pickle.dump(collect_scores, handle)

    with open(f'./seeds_gnn_2/surv_time_gnn_{seed}.pkl', 'wb') as handle:
        pickle.dump(collect_survival_time, handle)


