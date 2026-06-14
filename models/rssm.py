"""Recurrent State-Space Model (RSSM) world model — upgrade path.

A compact DreamerV3-style RSSM with a deterministic GRU recurrent state
and a stochastic latent. Kept lightweight so it can stand in for the MLP
`WorldModel` later. Exposes the same prediction heads (next_state, risk,
progress) plus latent transition utilities.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class RSSM(nn.Module):
    def __init__(self, state_dim=28, action_dim=4, hidden=256,
                 deter_dim=256, stoch_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # Encoder: observation -> features.
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # Recurrent transition.
        self.fc_input = nn.Linear(stoch_dim + action_dim, hidden)
        self.gru = nn.GRUCell(hidden, deter_dim)

        # Prior: from deterministic state -> stochastic latent params.
        self.fc_prior = nn.Sequential(nn.Linear(deter_dim, hidden), nn.ReLU())
        self.prior_mean = nn.Linear(hidden, stoch_dim)
        self.prior_std = nn.Linear(hidden, stoch_dim)

        # Posterior: from deterministic state + encoded obs -> latent params.
        self.fc_post = nn.Sequential(nn.Linear(deter_dim + hidden, hidden), nn.ReLU())
        self.post_mean = nn.Linear(hidden, stoch_dim)
        self.post_std = nn.Linear(hidden, stoch_dim)

        # Decoders / heads.
        feat = deter_dim + stoch_dim
        self.head_state = nn.Linear(feat, state_dim)
        self.head_risk = nn.Sequential(nn.Linear(feat, 1), nn.Sigmoid())
        self.head_progress = nn.Linear(feat, 1)

    # ------------------------------------------------------------------ #
    def initial_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.deter_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    @staticmethod
    def _dist(mean, std_logits):
        std = torch.nn.functional.softplus(std_logits) + 0.1
        return Normal(mean, std)

    def prior_step(self, deter, stoch, action):
        """Advance the latent one step without an observation (imagination)."""
        x = torch.cat([stoch, action], dim=-1)
        x = torch.relu(self.fc_input(x))
        deter = self.gru(x, deter)
        h = self.fc_prior(deter)
        dist = self._dist(self.prior_mean(h), self.prior_std(h))
        stoch = dist.rsample()
        return deter, stoch, dist

    def posterior_step(self, deter, obs):
        """Refine the latent using an observation."""
        enc = self.encoder(obs)
        h = self.fc_post(torch.cat([deter, enc], dim=-1))
        dist = self._dist(self.post_mean(h), self.post_std(h))
        stoch = dist.rsample()
        return stoch, dist

    def decode(self, deter, stoch):
        feat = torch.cat([deter, stoch], dim=-1)
        return (
            self.head_state(feat),
            self.head_risk(feat),
            self.head_progress(feat),
        )

    def forward(self, state, action, hidden_state=None):
        """One observe+predict step. Mirrors WorldModel.forward signature
        with an extra recurrent state, returning predictions and new hidden.
        """
        batch = state.shape[0]
        if hidden_state is None:
            deter, stoch = self.initial_state(batch, state.device)
        else:
            deter, stoch = hidden_state

        deter, stoch, _ = self.prior_step(deter, stoch, action)
        stoch, _ = self.posterior_step(deter, state)
        next_state_hat, risk_hat, progress_hat = self.decode(deter, stoch)
        return next_state_hat, risk_hat, progress_hat, (deter, stoch)
