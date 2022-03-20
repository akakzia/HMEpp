import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetworkFlat(nn.Module):
    def __init__(self, inp, out):
        super(QNetworkFlat, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out)

        # Q2 architecture
        self.linear4 = nn.Linear(inp, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicyFlat(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(GaussianPolicyFlat, self).__init__()

        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, out)
        self.log_std_linear = nn.Linear(256, out)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyFlat, self).to(device)