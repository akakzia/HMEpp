import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rl_modules.networks import GaussianPolicyFlat, QNetworkFlat, PhiNetwork

epsilon = 1e-6


class FlatCritic(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(FlatCritic, self).__init__()

        self.model = QNetworkFlat(dim_inp + 18, dim_out)
        self.phi_obs = PhiNetwork(6, 18)
        self.phi_g = PhiNetwork(2, 18)

    def forward(self, obs, act, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        # Edge features
        obs_z = self.phi_obs(obs)
        g_z = self.phi_g(g)

        # aggregation 
        agg_z = obs_z + g_z

        inp_state = torch.cat([obs, g, agg_z], dim=-1)

        q1_pi_tensor, q2_pi_tensor = self.model(inp_state, act)

        return q1_pi_tensor, q2_pi_tensor

class FlatActor(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(FlatActor, self).__init__()

        self.model = GaussianPolicyFlat(dim_inp + 18, dim_out)
        self.phi_obs = PhiNetwork(6, 18)
        self.phi_g = PhiNetwork(2, 18)


    def forward(self, obs, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        # Edge features
        obs_z = self.phi_obs(obs)
        g_z = self.phi_g(g)

        # aggregation 
        agg_z = obs_z + g_z

        # Readout
        inp = torch.cat([obs, g, agg_z], dim=-1)

        mean, logstd = self.model(inp)
        return mean, logstd

    def sample(self, obs, g):
        mean, log_std = self.forward(obs, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class FlatSemantic:
    def __init__(self, env_params):
        self.dim_obs = env_params['obs']
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        dim_actor_input = self.dim_obs + self.dim_goal
        dim_actor_output = self.dim_act

        dim_critic_input = self.dim_obs + self.dim_goal + self.dim_act
        dim_critic_output = 1

        self.critic = FlatCritic(dim_critic_input, dim_critic_output)
        self.critic_target = FlatCritic(dim_critic_input, dim_critic_output)
        self.actor = FlatActor(dim_actor_input, dim_actor_output)

    def policy_forward_pass(self, obs, g, no_noise=False):
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, g)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, g)

    def forward_pass(self, obs, g, actions=None):
        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, g)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, g)
            return self.critic.forward(obs, actions, g)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, g)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None