"""Trajectory data collection and processing for the traffic predictor.

``TrajectoryBuffer`` stores (history, future, agent_type) triples gathered from
rollouts; ``extract_agent_trajectories`` turns a recorded episode into such
triples by reading each VRU/vehicle slot out of the flat state vector and
forming sliding history->future windows; ``evaluate_prediction_accuracy``
reports ADE / FDE / success-rate.

State-vector layout (dim=28, see env/carla_env.py):
  ego (x, y) at 0, 1; nearest vehicle (speed, heading, rel_x, rel_y) at
  14, 15, 16, 17; VRU0 (dist, speed, heading, rel_x, rel_y) at 18..22;
  VRU1 at 23..27. Absolute agent position = ego position + relative offset.
"""
import numpy as np
import torch


# Agent slots: (relative_x_index, relative_y_index, speed_index, heading_index, class)
_AGENT_SLOTS = [
    (21, 22, 19, 20, 0),    # VRU0  -> pedestrian/cyclist (class 0)
    (26, 27, 24, 25, 0),    # VRU1  -> pedestrian/cyclist (class 0)
    (16, 17, 14, 15, 2),    # nearest vehicle (class 2)
]


class TrajectoryBuffer:
    """Stores agent trajectories (history, future, agent_type) for training."""

    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.histories = []
        self.futures = []
        self.types = []

    def add_trajectory(self, agent_id, history, future, agent_type):
        if len(self.histories) >= self.capacity:
            # Drop the oldest to stay within capacity.
            self.histories.pop(0)
            self.futures.pop(0)
            self.types.pop(0)
        self.histories.append(np.asarray(history, dtype=np.float32))
        self.futures.append(np.asarray(future, dtype=np.float32))
        self.types.append(int(agent_type))

    def _stack(self, idx):
        h = torch.as_tensor(np.stack([self.histories[i] for i in idx]))
        f = torch.as_tensor(np.stack([self.futures[i] for i in idx]))
        t = torch.as_tensor(np.asarray([self.types[i] for i in idx],
                                       dtype=np.int64))
        return h, f, t

    def get_batch(self, batch_size):
        """Random batch -> (histories, futures, types) tensors."""
        n = min(batch_size, len(self.histories))
        idx = np.random.choice(len(self.histories), size=n, replace=False)
        return self._stack(idx)

    def sample_hard(self, batch_size, difficulty=1.0):
        """Sample emphasising hard predictions (longer/faster futures).

        Each future's "hardness" is its total displacement; ``difficulty`` in
        [0, 1] interpolates from uniform (0) to fully hardness-weighted (1).
        """
        n_total = len(self.histories)
        if n_total == 0:
            return self._stack([])
        hardness = np.asarray([
            float(np.sum(np.linalg.norm(np.diff(f[:, :2], axis=0), axis=-1)))
            for f in self.futures
        ], dtype=np.float64)
        weights = (1.0 - difficulty) + difficulty * hardness
        total = weights.sum()
        probs = (weights / total) if total > 0 else None
        n = min(batch_size, n_total)
        idx = np.random.choice(n_total, size=n, replace=False, p=probs)
        return self._stack(idx)

    def clear(self):
        self.histories.clear()
        self.futures.clear()
        self.types.clear()

    def __len__(self):
        return len(self.histories)


def extract_agent_trajectories(env, episode_data, horizon=8):
    """Turn a recorded episode into (history, future, agent_type) triples.

    ``episode_data`` must contain ``'states'`` — a list/array of state vectors.
    For each agent slot we build the absolute-position time series and slide a
    ``seq_len`` history -> ``horizon`` future window over it.
    """
    seq_len = getattr(env, "history_length", 5)
    states = np.asarray(episode_data["states"], dtype=np.float32)
    if states.ndim != 2 or states.shape[0] < seq_len + horizon:
        return []

    out = []
    ego_xy = states[:, 0:2]
    for rx, ry, spd, hdg, cls in _AGENT_SLOTS:
        abs_x = ego_xy[:, 0] + states[:, rx]
        abs_y = ego_xy[:, 1] + states[:, ry]
        vx = states[:, spd] * np.cos(states[:, hdg])
        vy = states[:, spd] * np.sin(states[:, hdg])
        series = np.stack(
            [abs_x, abs_y, vx, vy, np.full_like(abs_x, cls)], axis=-1
        ).astype(np.float32)

        last = len(series) - seq_len - horizon + 1
        for i in range(last):
            history = series[i:i + seq_len]
            future = series[i + seq_len:i + seq_len + horizon]
            out.append((history, future, int(cls)))
    return out


def evaluate_prediction_accuracy(predictor, test_batch, horizon, success_thresh=1.0):
    """Compute ADE / FDE / success-rate for a predictor on a test batch.

    ``test_batch`` is ``(histories, futures, types)``. ADE is the mean L2
    displacement error over the horizon; FDE is the error at the final step;
    success-rate is the fraction of trajectories whose ADE is below
    ``success_thresh`` (metres).
    """
    histories, futures, _types = test_batch
    if len(histories) == 0:
        return {"ade": 0.0, "fde": 0.0, "success_rate": 0.0}

    pred_traj, _pred_vel, _unc = predictor(histories)
    target = torch.as_tensor(np.asarray(futures))[:, :horizon, :2].float().to(
        pred_traj.device)
    pred_traj = pred_traj[:, :horizon]

    disp = torch.linalg.norm(pred_traj - target, dim=-1)   # (B, horizon)
    ade_per = disp.mean(dim=1)                              # (B,)
    fde_per = disp[:, -1]                                   # (B,)
    success_rate = float((ade_per < success_thresh).float().mean().item())
    return {
        "ade": float(ade_per.mean().item()),
        "fde": float(fde_per.mean().item()),
        "success_rate": success_rate,
    }
