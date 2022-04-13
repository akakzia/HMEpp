import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
from bidict import bidict
import numpy as np
from rollout import HMERolloutWorker
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
from arguments import get_args
import networkit as nk
from graph.semantic_graph import SemanticGraph
from graph.semantic_network import SemanticNetwork

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps, 'cells': env.cells_graph}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahmed/models/'
    model_path = path + 'model_90.pt'

    args = get_args()

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward)
        policy.load(model_path)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = HMERolloutWorker(env, policy,  args)

    # Initialize SP graph
    sp_nk_graph = nk.Graph(0,weighted=True, directed=True)
    sp_semantic_graph = SemanticGraph(bidict(), sp_nk_graph, args=args)
    sp_network = SemanticNetwork(semantic_graph=sp_semantic_graph, args=args)

    # create network
    sp_network.create(args.env_params['cells'])

    # load agent graph
    nk_graph = nk.Graph(0, weighted=True, directed=True)
    semantic_graph = SemanticGraph(bidict(), nk_graph, args=args)
    agent_network = SemanticNetwork(semantic_graph, args=args)
    agent_network = agent_network.load(path, 90, args)

    # Affect teacher to agent
    agent_network.teacher.oracle_graph = sp_network.semantic_graph

    eval_goals = [(e[-2], e[-1]) for e in args.env_params['cells']]

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.test_social_rollouts(eval_goals, agent_network,
                                               episode_duration=args.episode_duration,
                                               animated=True)
        results = np.array([e['rewards'][-1].astype(np.float32) for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))

