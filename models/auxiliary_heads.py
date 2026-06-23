"""Auxiliary grounding objectives for the world model / representation.

Two prediction heads ground the latent state in semantics:
  * ``SceneReconstructionHead`` reconstructs a bird's-eye-view occupancy grid,
  * ``RiskDensityHead`` predicts a scalar scene-risk density,
and a ``WorldModelEnsemble`` gives epistemic uncertainty (model disagreement),
used by the S-DBS planner to scale search effort in ambiguous scenes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# Scene reconstruction (BEV occupancy)
# ---------------------------------------------------------------------- #
class SceneReconstructionHead(nn.Module):
    """Reconstructs a bird's-eye-view occupancy grid from the latent state."""

    def __init__(self, latent_dim, output_channels=1, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.output_channels = output_channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_channels * grid_size * grid_size),
        )

    def forward(self, latent):
        """Returns occupancy logits, shape (B, H_grid, W_grid)."""
        b = latent.shape[0]
        out = self.net(latent)
        return out.view(b, self.grid_size, self.grid_size)

    def loss(self, logits, target_occupancy):
        """Binary cross-entropy against a {0,1} occupancy target."""
        return F.binary_cross_entropy_with_logits(
            logits, target_occupancy.float()
        )


# ---------------------------------------------------------------------- #
# Scalar risk density
# ---------------------------------------------------------------------- #
class RiskDensityHead(nn.Module):
    """Predicts a scalar scene-risk density from the latent state."""

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, latent):
        """Returns risk density, shape (B,)."""
        return self.net(latent).squeeze(-1)

    def loss(self, pred, target_risk_density):
        return F.mse_loss(pred, target_risk_density.reshape(-1).float())


# ---------------------------------------------------------------------- #
# World-model ensemble (epistemic uncertainty)
# ---------------------------------------------------------------------- #
class WorldModelEnsemble(nn.Module):
    """An ensemble of world models for epistemic-uncertainty estimation."""

    def __init__(self, world_model_class, state_dim, action_dim,
                 n_models=3, hidden=256, device="cpu"):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            world_model_class(state_dim, action_dim, hidden)
            for _ in range(n_models)
        ])
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state, action):
        """Returns (mean_next_state, std_next_state, disagreement).

        ``disagreement`` is the per-sample mean predictive std, shape (B,).
        """
        preds = [m(state, action)[0] for m in self.models]
        stack = torch.stack(preds, dim=0)            # (n_models, B, state_dim)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        disagreement = std.mean(dim=-1)              # (B,)
        return mean, std, disagreement

    def loss(self, batch):
        """Mean MSE across all ensemble members (state + risk + progress)."""
        states = batch["states"].float().to(self.device)
        actions = batch["actions"].float().to(self.device)
        next_states = batch["next_states"].float().to(self.device)
        risk = batch["risk_targets"].reshape(-1).float().to(self.device)
        progress = batch["progress_targets"].reshape(-1).float().to(self.device)

        total = 0.0
        for m in self.models:
            ns, rk, pg = m(states, actions)
            total = (total
                     + F.mse_loss(ns, next_states)
                     + F.mse_loss(rk.squeeze(-1), risk)
                     + F.mse_loss(pg.squeeze(-1), progress))
        return total / self.n_models


def world_model_disagreement(state, action, ensemble):
    """Scalar epistemic uncertainty from an ensemble, squashed to [0, 1)."""
    with torch.no_grad():
        _, _, disagreement = ensemble(state, action)
    return float(np.tanh(float(disagreement.mean().item())))
