import numpy as np
import torch


class RolloutBuffer:
    """
    Stores exactly T timesteps of experience for one PPO iteration.
    After compute_returns_and_advantages() is called, get_minibatches()
    yields shuffled minibatches for K epochs of PPO updates.
    """

    def __init__(self, T, obs_dim, action_dim, gamma, lam, device):
        self.T = T
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self.states    = np.zeros((T, obs_dim),    dtype=np.float32)
        self.actions   = np.zeros((T, action_dim), dtype=np.float32)
        self.rewards   = np.zeros(T,               dtype=np.float32)
        self.log_probs = np.zeros(T,               dtype=np.float32)
        self.values    = np.zeros(T,               dtype=np.float32)
        self.dones     = np.zeros(T,               dtype=np.float32)

        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns    = np.zeros(T, dtype=np.float32)

        self.ptr = 0

    def store(self, state, action, reward, log_prob, value, done):
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def is_full(self):
        return self.ptr == self.T

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        GAE backward pass — equation (11) from the PPO paper.

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t     = delta_t + (gamma * lambda) * A_{t+1} * (1 - done_{t+1})
        """
        gae = 0.0
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_value = last_value
                next_done  = last_done
            else:
                next_value = self.values[t + 1]
                next_done  = self.dones[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1.0 - next_done) - self.values[t]
            gae   = delta + self.gamma * self.lam * (1.0 - next_done) * gae

            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_minibatches(self, minibatch_size):
        """Yield shuffled minibatches as device tensors. Call once per epoch."""
        assert self.ptr == self.T, "Buffer must be full before generating minibatches"

        indices    = np.random.permutation(self.T)
        states     = torch.FloatTensor(self.states).to(self.device)
        actions    = torch.FloatTensor(self.actions).to(self.device)
        log_probs  = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.advantages).to(self.device)
        returns    = torch.FloatTensor(self.returns).to(self.device)

        for start in range(0, self.T, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield (
                states[idx],
                actions[idx],
                log_probs[idx],
                advantages[idx],
                returns[idx],
            )

    def reset(self):
        self.ptr = 0
