import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func, args):
        self.env_params = env_params
        self.T = args.episode_duration
        self.size = buffer_size // self.T

        # memory management
        self.sample_func = sample_func

        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                       'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                       'ag_continuous': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                       'g': np.empty([self.size, self.T, self.env_params['goal']]),
                       'actions': np.empty([self.size, self.T, self.env_params['action']]),

                       }

        self.goal_ids = np.zeros([self.size])  # contains id of achieved goal (discovery rank)
        self.goal_ids.fill(np.nan)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(episode_batch):
                # store the informations
                self.buffer['obs'][idxs[i]] = e['obs']
                self.buffer['ag'][idxs[i]] = e['ag']
                self.buffer['ag_continuous'][idxs[i]] = e['ag_continuous']
                self.buffer['g'][idxs[i]] = e['g']
                self.buffer['actions'][idxs[i]] = e['act']

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            # Randomly select episodes from buffer (without reference to the goal class)
            for key in self.buffer.keys():
                temp_buffers[key] = self.buffer[key][:self.current_size]


        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        temp_buffers['ag_next_continuous'] = temp_buffers['ag_continuous'][:, 1:, :]


        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx