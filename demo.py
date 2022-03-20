import numpy as np
import gym
import env
import networkit as nk
from bidict import bidict
from graph.semantic_network import SemanticNetwork
from graph.semantic_graph import SemanticGraph
from arguments import get_args

# Get parameters
args = get_args()

env_name = 'PointUMaze-v1'

env = gym.make(env_name)

env.reset()

d = False

cells = env.cells_graph

nk_graph = nk.Graph(0,weighted=True, directed=True)
semantic_graph = SemanticGraph(bidict(), nk_graph, args=args)
sp_network = SemanticNetwork(semantic_graph=semantic_graph, args=args)

# create network
sp_network.create(cells)

# Check graph
# for node in sp_network.semantic_graph.configs:
#     print('Source node: {}'.format(node))
#     for n in sp_network.semantic_graph.iterNeighbors(node):
#         print(n)
#     print('----------------------------')

for node in sp_network.semantic_graph.configs:
    print(node)
    obs = env.reset_goal(goal=(-3, -3))
    for _ in range(100):
        action = env.action_space.sample()
        obs, r, d, _ = env.step(action)
        env.render()
    stop = 1
env.close()