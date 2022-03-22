import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import HMERolloutWorker
from utils import init_storage
import time
from mpi_utils import logger
import networkit as nk
from typing import DefaultDict
from bidict import bidict
from graph.semantic_network import SemanticNetwork
from graph.semantic_graph import SemanticGraph

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps, 'cells': env.cells_graph}
    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    assert args.env_name in ['PointHard-v1', 'PointUMaze-v1', 'PointSquareRoom-v1', 'PointCorridor-v1', 'PointLongCorridor-v1', 'Point4Rooms-v1', 'PointBottleneck-v1'], \
    'Please use one of the following environments: PointHard-v1, PointUMaze-v1, PointSquareRoom-v1, PointCorridor-v1, PointLongCorridor-v1, Point4Rooms-v1'
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    args.env_params = get_env_params(env)

    # Initialize RL Agent
    assert args.agent in ['SAC', 'TQC'], 'Please select SAC or TQC as algorithm'
    policy = RLAgent(args, env.compute_reward)

    # Initialize Rollout Worker
    rollout_worker = HMERolloutWorker(env, policy,  args)

    # Create graphs
    agent_nk_graph = nk.Graph(0,weighted=True, directed=True)
    sp_nk_graph = nk.Graph(0,weighted=True, directed=True)
    sp_semantic_graph = SemanticGraph(bidict(), sp_nk_graph, args=args)
    agent_semantic_graph = SemanticGraph(bidict(), agent_nk_graph, args=args)
    sp_network = SemanticNetwork(semantic_graph=sp_semantic_graph, args=args)
    agent_network = SemanticNetwork(semantic_graph=agent_semantic_graph, args=args)

    # create network
    sp_network.create(args.env_params['cells'])

    # Affect teacher to agent
    agent_network.teacher.oracle_graph = sp_network.semantic_graph

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        # time_dict = dict(goal_sampler=0,
        #                  rollout=0,
        #                  gs_update=0,
        #                  store=0,
        #                  norm_update=0,
        #                  policy_train=0,
        #                  eval=0,
        #                  epoch=0)
        time_dict = DefaultDict(int)


        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):
            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.train_rollout(agent_network= agent_network,
                                                    time_dict=time_dict)
            time_dict['rollout'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Agent Network Update : 
            t_i = time.time()
            agent_network.update(episodes)
            time_dict['update_graph'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += args.num_rollouts_per_mpi * args.num_workers

        time_dict['lp_update'] += time.time() - t_i
        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = [(e[-2], e[-1]) for e in args.env_params['cells']]
            episodes = rollout_worker.test_rollout(eval_goals,agent_network,
                                                            episode_duration=args.episode_duration,
                                                            animated=False)
            # results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
            rewards = np.array([e['rewards'][-1] for e in episodes])
            # rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(rewards, root=0)
            # all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                # assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                agent_network.log(logger)
                time_dict['episode_count'] += episode_count
                time_dict['global_sr'] += global_sr

                for g_id in np.arange(1, len(av_res) + 1):
                    time_dict['Eval_SR_{}'.format(g_id)] += av_res[g_id-1]
                log_and_save(time_dict)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    agent_network.save(model_path,epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save(time_dict):
    # goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict)
    for k, l in time_dict.items():
        logger.record_tabular(k, l)
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)