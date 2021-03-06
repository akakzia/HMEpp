import numpy as np


def compute_her_rewards(ag, g):
    """ computes binary reward given achieved and desired cell """
    reward = 1.0 if np.linalg.norm(ag - g) <= 0.9 else 0
    return reward

class her_sampler:
    def __init__(self, args, reward_func=None):
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        self.future_p = 1 - (1. / (1 + args.replay_k))
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        if self.replay_strategy == 'final':
            # fictive goal is the final achieved goal of the selected HER episodes
            future_ag = episode_batch['ag_continuous'][episode_idxs[her_indexes],-1]
        else:
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # replace goal with achieved goal
            future_ag = episode_batch['ag_continuous'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next_continuous'],
        #                                                                                     transitions['g'])]), 1)
        transitions['r'] = np.expand_dims(np.array([compute_her_rewards(ag_next, g) for ag_next, g in zip(transitions['ag_next_continuous'],
                                                                                            transitions['g'])]), 1)

        return transitions