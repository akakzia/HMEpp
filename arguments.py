import argparse
import numpy as np
from mpi4py import MPI


"""
Here are the param for the training
"""


def get_args():
    parser = argparse.ArgumentParser()
    # the general arguments
    parser.add_argument('--seed', type=int, default=np.random.randint(1e5), help='random seed')
    parser.add_argument('--num-workers', type=int, default=MPI.COMM_WORLD.Get_size(), help='the number of cpus to collect samples')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    # the environment arguments
    parser.add_argument('--env-name', type=str, default='PointCorridor-v1', help='The name of the Maze environment')
    parser.add_argument('--agent', type=str, default='SAC', help='the RL algorithm name')
    # the training arguments
    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=10, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=30, help='the times to update the network')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=25, help='the rollouts per mpi')
    parser.add_argument('--episode-duration', type=int, default=20, help='number of time steps for each mini episodes')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    # the replay arguments
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    # The RL arguments
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--alpha', type=float, default=0.02, help='entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, help='Tune entropy')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-entropy', type=float, default=0.001, help='the learning rate of the entropy')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--freq-target_update', type=int, default=1, help='the frequency of updating the target networks')
    # the output arguments
    parser.add_argument('--evaluations', type=bool, default=True, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='output/', help='the path to save the models')
    # the memory arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    # the preprocessing arguments
    parser.add_argument('--clip-obs', type=float, default=20, help='the clip ratio')
    parser.add_argument('--normalize_goal', type=bool, default=False, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--clip-range', type=float, default=20, help='the clip range')
    # the gnns arguments
    parser.add_argument('--architecture', type=str, default='flat', help='The architecture of the networks')
    # the testing arguments
    parser.add_argument('--n-test-rollouts', type=int, default=5, help='the number of tests')
    # graph arguments : 
    parser.add_argument('--edge-sr', type=str, default='exp_moving_average', help='moving_average or exp_moving_average')
    parser.add_argument('--edge-lr', type=float, default=0.01, help='SR learning rate')
    parser.add_argument('--edge-prior', type=float, default=0.5, help='default value for edges')
    parser.add_argument('--expert-graph-start', type=bool, default=False, help='If the agent starts with an expert graph')
    parser.add_argument('--evaluation-algorithm', type=str, default='dijkstra', help='dijkstra (best SR) or bfs (shortest path)')
    # rollout exploration args
    parser.add_argument('--rollout-exploration', type=str, default='sr_and_k_distance', help='method to compute best path in train rollouts : sr_and_best_distance sr_and_k_distance or sample_sr')
    parser.add_argument('--rollout-exploration-k', type=int, default=5, help='sample among k best paths')
    parser.add_argument('--rollout-distance-ratio', type=float, default=0.5, help='indicate the ratio at which exploration alternate beetween sr and distance criteria')
    parser.add_argument('--max-path-len', type=int, default=25, help='maximum path length')
    # Help Me Explore args
    parser.add_argument('--intervention-prob', type=float, default=1., help='the probability of SP intervention')
    parser.add_argument('--exploration-noise-prob', type=float, default=0., help='When going to frontier, apply noise at ratio')
    parser.add_argument('--strategy', type=int, default=2, help='Possible values: 0: Frontier; 1: Frontier and Stop, 2: Frontier and Beyond'
                                                                   '3: Beyond')

    parser.add_argument('--internalization-prob', type=float, default=0.0, help='the probability of internalizing SP intervention')
    parser.add_argument('--ss-internalization-prob', type=float, default=0., help='the probability of internalizing stepping stones')

    parser.add_argument('--teacher-bias', type=bool, default=True, help='If True, automatically add given goals by SP to the agent graph')

    args = parser.parse_args()

    return args