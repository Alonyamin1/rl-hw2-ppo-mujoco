import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, Critic


class PPOAgent:

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        lr=3e-4,
        clip_eps=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.device        = device
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        # One optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    @torch.no_grad()
    def select_action(self, state):
        """
        Sample an action from the current policy.
        Returns numpy action, scalar log_prob, scalar value.
        """
        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist     = self.actor.get_distribution(state_t)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value    = self.critic(state_t)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def get_value(self, state):
        """V(s) for the last state after rollout ends. Used for GAE bootstrap."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.critic(state_t).item()

    def update(self, buffer, n_epochs, minibatch_size):
        """
        Run K epochs of PPO updates on the data in buffer.
        Returns dict of mean losses for logging.
        """
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(n_epochs):
            for states, actions, old_log_probs, advantages, returns in buffer.get_minibatches(minibatch_size):

                # Policy loss (clipped surrogate) 
                new_log_probs = self.actor.get_log_prob(states, actions)
                entropy       = self.actor.get_entropy(states).mean()

                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss 
                values     = self.critic(states)
                value_loss = nn.functional.mse_loss(values, returns)

                # Combined loss — equation (9)
                # Minimize: -L_CLIP + c1*L_VF - c2*Entropy
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                n_updates         += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
        }

    def save(self, path):
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
