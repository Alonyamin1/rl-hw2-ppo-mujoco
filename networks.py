import torch
import torch.nn as nn
import numpy as np


def init_weights(layer, std=np.sqrt(2), bias=0.0):
    """Orthogonal init — standard practice for PPO."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)


class Actor(nn.Module):
    """
    Outputs the mean of a Gaussian policy.
    log_std is a separate learnable parameter (not a network output).
    Architecture: 2 hidden layers of 64 units, tanh activations (as in the paper).
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.net.apply(init_weights)
        init_weights(self.net[-1], std=0.01)

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        mean = self.forward(x)
        std  = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def get_log_prob(self, x, actions):
        dist = self.get_distribution(x)
        return dist.log_prob(actions).sum(dim=-1)

    def get_entropy(self, x):
        dist = self.get_distribution(x)
        return dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    """Outputs V(s) — a scalar value estimate."""

    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.net.apply(init_weights)
        init_weights(self.net[-1], std=1.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)
