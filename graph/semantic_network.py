import random
import numpy as np
from graph.semantic_graph import SemanticGraph
from mpi4py import MPI
from graph.teacher import Teacher
import pickle

EPSILON_TRAVEL = 0.1

def get_consecutive_pairs(ags):
    """ Given a sequence of ags for an episode, returns a list of consecutive pairs """
    res = set()
    i = 0
    for i in range(ags.shape[0]-1):
        if tuple(ags[i]) != tuple(ags[i+1]):
            res.add((tuple(ags[i]), tuple(ags[i+1])))
    
    return res

class SemanticNetwork():
    
    def __init__(self,semantic_graph :SemanticGraph,exp_path=None,args=None):
        self.teacher = Teacher(args)
        self.semantic_graph = semantic_graph
        self.args = args
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.exp_path = exp_path

        self.nb_discovered_goals = 0
        # Keep track of successes and failures to estimate Learning Progress
        self.successes_and_failures = []
        self.queue_lp = 100
        self.LP = 0
        self.LP_max = 0.5

        # Keep track of number of discovered goals to estimate discovery speed
        self.nb_discovered_list = []
        self.queue_discovery = 10 * args.num_workers
        self.DS = 0

        # Query proba initialization
        self.query_lp = 0
        self.query_ds = 0
        self.query_p = 0

        # Keep counts of edge travels
        self.edge_travel_counts = {}

        self.init_stats()

    def create(self, cells):
        """ Given a list of cells of the form [col_index, row_index, x, y], creates a directed semantic graph """
        indexes_list = []
        # Create all nodes
        for cell in cells:
            indexes_list.append((cell[0], cell[1]))
            self.semantic_graph.create_node((cell[2], cell[3]))    
        
        # Create all edges
        for i, indexes in enumerate(cells):
            for j, moving_indexes in enumerate(cells):
                if (((abs(indexes[0] - moving_indexes[0]) == 1) and (abs(indexes[1] - moving_indexes[1]) == 0)) or 
                ((abs(indexes[1] - moving_indexes[1]) == 1) and (abs(indexes[0] - moving_indexes[0]) == 0))) and (i != j):
                    self.update_or_create_edge((cells[i][2], cells[i][3]), (cells[j][2], cells[j][3]), 1)
                    self.update_or_create_edge((cells[j][2], cells[j][3]), (cells[i][2], cells[i][3]), 1)
        

    def update(self,episodes):
        all_episodes = MPI.COMM_WORLD.allgather(episodes)
        all_episode_list = [e for eps in all_episodes 
                                for e in eps] # flatten the list of episodes gathered by all actors
        # update agent graph :
        for e in all_episode_list:
            # For each episode, create an edge between all consecutive ag
            consecutive_pairs_set = get_consecutive_pairs(e['ag'])
            if len(consecutive_pairs_set) > 0:
                for pair in consecutive_pairs_set:
                    # Create node for each semantic state
                    self.semantic_graph.create_node(pair[0])
                    self.semantic_graph.create_node(pair[1])

                    # Create edge if not existant using edge prior
                    if not self.semantic_graph.hasEdge(pair[0], pair[1]):
                        self.semantic_graph.create_edge_stats((pair[0], pair[1]), self.args.edge_prior)
                    
                    # Update travel counts for Neighbour strategy
                    self.update_travel_counts(pair[0], pair[1])
            
            # If the only one consecutive pair, then append success estimates
            # if len(consecutive_pairs_set) == 1:
            #     goal = tuple(e['g'][-1])
            #     success = e['success'][-1]
            #     if not e['beyond_fail'] and self.semantic_graph.getNodeId(goal) is not None:
            #         self.update_or_create_edge(pair[0], goal, success)
                
            if e['self_eval']:
                self.successes_and_failures.append(e['success'][-1])


            # condition = (str(e['ag'][-1]) == str(e['ag'][-2]))
            # if condition:
            #     start_config = tuple(e['ag'][0])
            #     achieved_goal = tuple(e['ag'][-1])
            #     goal = tuple(e['g'][-1])
            #     success = e['success'][-1]

            #     self.semantic_graph.create_node(start_config)
            #     self.semantic_graph.create_node(achieved_goal)

            #     if not e['beyond_fail']: 
            #         if self.semantic_graph.getNodeId(goal) is not None:
            #             self.update_or_create_edge(start_config, goal, success)
            #     if (achieved_goal != goal and start_config != achieved_goal
            #             and not self.semantic_graph.hasEdge(start_config, achieved_goal)):
            #         self.semantic_graph.create_edge_stats((start_config, achieved_goal), self.args.edge_prior)
            
            # if e['self_eval']:
            #     self.successes_and_failures.append(e['success'][-1])

        # update frontier :
        self.semantic_graph.update()
        self.teacher.compute_frontier(self.semantic_graph)
        self.nb_discovered_goals = len(self.semantic_graph.configs)
        self.nb_discovered_list.append(self.nb_discovered_goals)
        self.update_estimates()
    
    def update_estimates(self):
        """ Updates estimates of learning progress and discovery speed to determine query proba """
        self.successes_and_failures = self.successes_and_failures[-self.queue_lp:]
        self.nb_discovered_list = self.nb_discovered_list[-self.queue_discovery:]
        # Update LP estimate
        n_points_lp = len(self.successes_and_failures)
        if n_points_lp > 20:
            sf = np.array(self.successes_and_failures)
            self.LP = np.abs(np.sum(sf[n_points_lp // 2:]) - np.sum(sf[: n_points_lp // 2])) / n_points_lp
        
        # Update DS estimate 
        n_points_ds = len(self.nb_discovered_list)
        if n_points_ds > 10:
            nb_disc = np.array(self.nb_discovered_list)
            self.DS = (nb_disc[-1] - nb_disc[-n_points_ds]) / nb_disc[-1]
        
        # self.query_p = 0.5 * (((self.LP_max - self.LP) / self.LP_max) + self.DS)
        if n_points_lp > 20 and n_points_ds > 5:
            if self.LP == 0:
                self.query_lp = 1.
            else:
                self.query_lp = 1 / (500 * self.LP)
            if self.DS == 0:
                self.query_ds = 1.
            else:
                self.query_ds = 1 / (200 * self.DS)
            
            self.query_p = 0.5 * (self.query_ds + self.query_lp)

    def update_travel_counts(self, source, target):
        """ Given a traveled edge source -> target, update traveling counts """
        try:
            self.edge_travel_counts[(source, target)] += 1
        except KeyError:
            self.edge_travel_counts[(source, target)] = 1
    
    def update_or_create_edge(self,start,end,success):
        if start != end:
            if not self.semantic_graph.hasEdge(start,end):
                self.semantic_graph.create_edge_stats((start,end),self.args.edge_prior)
            self.semantic_graph.update_edge_stats((start,end),success)

    def get_path(self,start,goal,algorithm='dijkstra'):
        if self.args.expert_graph_start: 
            return self.teacher.oracle_graph.sample_shortest_path(start,goal,algorithm=algorithm)
        else : 
            return self.semantic_graph.sample_shortest_path(start,goal,algorithm=algorithm)

    def get_path_from_coplanar(self,target):
        if self.args.expert_graph_start : 
            return self.teacher.oracle_graph.get_path_from_coplanar(target)
        else : 
            return self.semantic_graph.get_path_from_coplanar(target)

    def sample_goal_uniform(self,nb_goal,use_oracle=True):
        if use_oracle:
            return self.teacher.sample_goal_uniform(nb_goal)
        else :
            try:
                return random.choices(self.semantic_graph.configs.inverse,k=nb_goal)
            except KeyError:
                # If there are no goals discovered yet, returns (0, 0)
                return [(0, 0) for _ in range(nb_goal)]

    def sample_goal_in_frontier(self,current_node,k):
        return self.teacher.sample_in_frontier(current_node,self.semantic_graph,k)
    
    def sample_terminal(self,current_node,k):
        return self.teacher.sample_terminal(current_node,self.semantic_graph,k)
    
    def sample_from_frontier(self,frontier_node,k):
        return self.teacher.sample_from_frontier(frontier_node,self.semantic_graph,k)
    
    def sample_many_from_frontier(self,frontier_node,k):
        return self.teacher.sample_many_from_frontier(frontier_node,self.semantic_graph,k)

    def sample_rand_neighbour(self,source,excluding = []):
        neighbours = list(filter( lambda x : x not in excluding, self.semantic_graph.iterNeighbors(source)))
        if neighbours:
            return random.choice(neighbours)
        else : 
            return None
    
    def sample_neighbour_by_counts(self,source, excluding = []):
        neighbours = list(filter( lambda x : x not in excluding, self.semantic_graph.iterNeighbors(source)))
        if neighbours:
            # Perform epsilon exploration
            # With proba epsilon, take a random neighbour, else, take the least traveled edge
            if np.random.uniform() < EPSILON_TRAVEL:
                return random.choice(neighbours), [source]
            else:
                least_traveled = neighbours[0]
                min_counts = self.edge_travel_counts[(source, least_traveled)]
                for i in range(1, len(neighbours)):
                    current_min_counts = self.edge_travel_counts[(source, neighbours[i])] 
                    if current_min_counts < min_counts:
                        least_traveled = neighbours[i]
                        min_counts = current_min_counts
                return least_traveled, neighbours + [source]
        else : 
            return None, None

    def sample_neighbour_based_on_SR_to_goal(self,source,reversed_dijkstra,goal, excluding = []):

        neighbors = [ n for n  in self.semantic_graph.iterNeighbors(source) if n not in excluding]

        if len(neighbors)>0:
            _,source_sr,_ = self.semantic_graph.sample_shortest_path_with_sssp(source,goal,reversed_dijkstra,reversed=True)

            source_to_neighbors_sr,neighbors_to_goal_sr,_ = self.semantic_graph.get_neighbors_to_goal_sr(source,neighbors,goal,reversed_dijkstra)
            source_to_neighbour_to_goal_sr = source_to_neighbors_sr*neighbors_to_goal_sr
            
            # remove neighbors with SR lower than current node :
            inds = neighbors_to_goal_sr>source_sr
            neighbors = np.array(neighbors)[inds]
            source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[inds]

            # filter neighbors :
            # Among multiple neighbors belonging to the same unordered edge, only keep one by sampling among highest SR neighbor_to_goal
            edges = [self.semantic_graph.edge_config_to_edge_id((source,tuple(neigh)))for neigh in neighbors]
            edges,inv_ids =  np.unique(np.array(edges),return_inverse = True)
            filtered_ids = np.empty_like(edges)
            for i,e in enumerate(edges):
                e_neigh_ids = np.where(inv_ids == i)[0]
                e_sr = neighbors_to_goal_sr[e_neigh_ids] 
                highest_neighbors_ids = e_neigh_ids[np.argwhere(e_sr == np.amax(e_sr)).flatten()]
                choosen_neighbor_id = np.random.choice(highest_neighbors_ids)
                filtered_ids[i] = choosen_neighbor_id
            neighbors = neighbors[filtered_ids]
            source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[filtered_ids]
            
            # only keep k_ largest probs :
            if len(source_to_neighbour_to_goal_sr) > self.args.rollout_exploration_k:
                inds = np.argpartition(source_to_neighbour_to_goal_sr, -self.args.rollout_exploration_k)[-self.args.rollout_exploration_k:]
                neighbors = np.array(neighbors)[inds]
                source_to_neighbour_to_goal_sr = source_to_neighbour_to_goal_sr[inds]
            sr_sum = np.sum(source_to_neighbour_to_goal_sr)
            if sr_sum == 0 :
                return None
            else : 
                probs = source_to_neighbour_to_goal_sr/sr_sum
                neighbour_id = np.random.choice(range(len(neighbors)),p = probs)
                return tuple(neighbors[neighbour_id])
        else : 
            return None

    def log(self,logger):
        self.semantic_graph.log(logger)
        # TODO : , à change selon qu'on soit unordered ou pas.
        logger.record_tabular('frontier_len',len(self.teacher.agent_frontier))
        logger.record_tabular('stepping_stones_len', len(self.teacher.agent_stepping_stones))
        logger.record_tabular('terminal_len', len(self.teacher.agent_terminal))
        logger.record_tabular('_LP', self.LP)
        logger.record_tabular('_DS', self.DS)
        logger.record_tabular('_query_proba', self.query_p)

    def save(self,model_path, epoch):
        self.semantic_graph.save(model_path+'/',f'{epoch}')
        with open(f"{model_path}/frontier_{epoch}.config", 'wb') as f:
            pickle.dump(self.teacher.agent_frontier,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(self, model_path,epoch,args) ->'SemanticNetwork':
        semantic_graph = SemanticGraph.load(model_path,f'{epoch}')
        with open(f"{model_path}frontier_{epoch}.config", 'rb') as f:
            frontier = pickle.load(f)
        agent_network = SemanticNetwork(semantic_graph,None,args)
        agent_network.teacher.agent_frontier = frontier
        return agent_network

    def sync(self):
        self.teacher.agent_frontier = MPI.COMM_WORLD.bcast(self.teacher.agent_frontier, root=0)
        if self.rank == 0:
            self.semantic_graph.save(self.exp_path+'/','temp')

        MPI.COMM_WORLD.Barrier()
        if self.rank!=0:
            self.semantic_graph = SemanticGraph.load(self.exp_path+'/','temp')

    def init_stats(self):
        self.stats = dict()
        for i in range(11):
            self.stats[i+1] = 0