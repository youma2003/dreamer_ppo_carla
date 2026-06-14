"""PPO Actor-Critic policy with a continuous (Box) action space.

Action layout: [steering, throttle, brake, stop_continue]
  steering   -> tanh  -> [-1, 1]
  throttle   -> sigmoid -> [0, 1]
  brake      -> sigmoid -> [0, 1]
  stop_cont. -> sigmoid -> [0, 1]

A diagonal Gaussian is defined in an unbounded "raw" space; the bounded
action is obtained by squashing. log-probs are computed in raw space so
that PPO ratios stay consistent between rollout collection and update.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim=28, action_dim=4, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.critic = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _squash(raw):
        """Map raw Gaussian samples to the bounded action space."""
        steering = torch.tanh(raw[..., 0:1])
        rest = torch.sigmoid(raw[..., 1:4])
        return torch.cat([steering, rest], dim=-1)

    def forward(self, state):
        h = self.trunk(state)
        mean = self.actor_mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def act(self, state):
        """Sample an action for rollout collection.

        Returns (bounded_action, log_prob, value), all detached.
        """
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        action = self._squash(raw_action)
        return action.detach(), log_prob.detach(), value.detach(), raw_action.detach()

    def evaluate(self, state, raw_action):
        """Evaluate stored raw actions for a PPO update.

        Returns (log_prob, entropy, value).
        """
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
