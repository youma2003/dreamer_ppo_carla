"""Trainer for the traffic / pedestrian trajectory predictor.

Collects agent trajectories from rollouts into a ``TrajectoryBuffer`` and trains
``TrafficPredictor`` on history -> future pairs with a position + velocity +
uncertainty loss (high-error steps are pushed to report higher uncertainty).
"""
import numpy as np
import torch

from utils.trajectory_utils import (
    TrajectoryBuffer, extract_agent_trajectories, evaluate_prediction_accuracy,
)


class TrafficPredictionTrainer:
    def __init__(self, predictor, config, device="cpu"):
        self.predictor = predictor
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            predictor.parameters(), lr=config.lr_tp, weight_decay=1e-4)
        self.trajectory_buffer = TrajectoryBuffer(capacity=50_000)
        self.min_ready = int(getattr(config, "tp_min_ready", 1000))

    # ------------------------------------------------------------------ #
    def collect_trajectories(self, env, num_episodes=100, max_steps=200):
        """Run episodes with random actions and fill the trajectory buffer."""
        horizon = self.predictor.horizon
        for _ in range(num_episodes):
            obs = env.reset()
            states = [np.asarray(obs, dtype=np.float32)]
            for _ in range(max_steps):
                action = np.random.uniform(
                    -1, 1, size=env.action_dim).astype(np.float32)
                obs, _r, done, _info = env.step(action)
                states.append(np.asarray(obs, dtype=np.float32))
                if done:
                    break
            for history, future, agent_type in extract_agent_trajectories(
                    env, {"states": states}, horizon=horizon):
                self.trajectory_buffer.add_trajectory(
                    None, history, future, agent_type)

    # ------------------------------------------------------------------ #
    def update(self, batch):
        """One gradient step on a (histories, futures, types) batch."""
        histories, futures, _types = batch
        histories = torch.as_tensor(np.asarray(histories)).float().to(self.device)
        futures = torch.as_tensor(np.asarray(futures)).float().to(self.device)
        horizon = self.predictor.horizon

        target_traj = futures[:, :horizon, :2]
        target_vel = futures[:, :horizon, 2:4]

        pred_traj, pred_vel, uncertainty = self.predictor(histories)

        loss_traj = torch.nn.functional.mse_loss(pred_traj, target_traj)
        loss_vel = torch.nn.functional.mse_loss(pred_vel, target_vel)

        # Uncertainty regression: weight high-error steps so the model learns to
        # report larger uncertainty exactly where it is wrong.
        error = (pred_traj - target_traj).abs().detach()
        threshold = float(getattr(self.config, "prediction_threshold", 0.1))
        unc_weight = torch.where(error > threshold,
                                 torch.ones_like(error),
                                 torch.full_like(error, 0.1))
        loss_unc = (unc_weight * (error - uncertainty) ** 2).mean()

        loss_total = loss_traj + 0.5 * loss_vel + 0.2 * loss_unc

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            disp = torch.linalg.norm(pred_traj - target_traj, dim=-1)
            ade = float(disp.mean().item())
            fde = float(disp[:, -1].mean().item())
        return {"loss": float(loss_total.item()), "ade": ade, "fde": fde}

    # ------------------------------------------------------------------ #
    def evaluate(self, test_batch, n_samples=1000):
        return evaluate_prediction_accuracy(
            self.predictor, test_batch, self.predictor.horizon)

    def is_ready(self):
        return len(self.trajectory_buffer) > self.min_ready
