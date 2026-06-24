"""Multi-agent trajectory prediction for VRUs and vehicles.

``TrafficPredictor`` is an LSTM encoder + MLP decoder that forecasts the next
``horizon`` positions / velocities (and an epistemic-uncertainty estimate) of a
single agent from a short history of ``[x, y, vx, vy, class]`` observations.
``MultiAgentPredictor`` wraps it to track and predict several agents at once and
to score collision risk between the ego plan and the predicted futures.

In mock mode the agent histories are essentially noise, so predictions are not
meaningful — but the full pipeline (tracking -> prediction -> collision-aware
planning) runs end to end, which is what the mock tests exercise.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# Collision-risk helper (shared by the planner and MultiAgentPredictor)
# ---------------------------------------------------------------------- #
def compute_collision_risk(ego_traj, agent_predictions, sigma=1.5):
    """Probability-like collision risk in [0, 1] between an ego trajectory and
    predicted agent trajectories.

    ``ego_traj``: array (T, 2) of ego positions.
    ``agent_predictions``: dict ``agent_id -> traj`` or ``agent_id -> (traj,
    uncertainty)`` where ``traj`` is (H, 2). Risk is the max over agents and
    matched timesteps of ``exp(-d^2 / (2 sigma^2))``; coincident positions give
    risk ~1.
    """
    ego = np.asarray(ego_traj, dtype=np.float32).reshape(-1, 2)
    risk = 0.0
    for pred in agent_predictions.values():
        traj = pred[0] if isinstance(pred, (tuple, list)) else pred
        traj = np.asarray(traj, dtype=np.float32).reshape(-1, 2)
        t = min(len(ego), len(traj))
        for i in range(t):
            d = float(np.linalg.norm(ego[i] - traj[i]))
            risk = max(risk, float(np.exp(-(d * d) / (2.0 * sigma * sigma))))
    return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------- #
# TrafficPredictor (single agent)
# ---------------------------------------------------------------------- #
class TrafficPredictor(nn.Module):
    def __init__(self, state_dim, agent_state_dim=5, horizon=8,
                 hidden_dim=128, max_speed=15.0, pos_bound=100.0, device="cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.agent_state_dim = agent_state_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.max_speed = max_speed
        self.pos_bound = pos_bound
        self.device = torch.device(device)

        self.encoder = nn.LSTM(agent_state_dim, hidden_dim, batch_first=True)
        self.head_traj = nn.Linear(hidden_dim, horizon * 2)
        self.head_vel = nn.Linear(hidden_dim, horizon * 2)
        self.head_unc = nn.Linear(hidden_dim, horizon * 2)

        self.to(self.device)

    def forward(self, agent_history):
        """agent_history: (B, seq_len, agent_state_dim).

        Returns (pred_traj, pred_vel, uncertainty), each (B, horizon, 2).
        """
        if not torch.is_tensor(agent_history):
            agent_history = torch.as_tensor(
                np.asarray(agent_history, dtype=np.float32))
        agent_history = agent_history.float().to(self.device)
        if agent_history.dim() == 2:                 # (seq, dim) -> (1, seq, dim)
            agent_history = agent_history.unsqueeze(0)

        _, (h_n, _) = self.encoder(agent_history)
        h = h_n[-1]                                   # (B, hidden_dim)
        b = h.shape[0]

        # Predict bounded *displacement* from the last observed position
        # (residual prediction): far easier to learn than absolute coordinates,
        # while the final positions stay bounded by last_pos +- pos_bound.
        last_pos = agent_history[:, -1, 0:2]          # (B, 2)
        delta = (torch.tanh(self.head_traj(h)) * self.pos_bound).view(
            b, self.horizon, 2)
        pred_traj = last_pos.unsqueeze(1) + delta

        # Velocity is bounded to [-max_speed, max_speed] with tanh (signed
        # components, starting near zero) rather than sigmoid: it matches real
        # vx/vy which can be negative and avoids a large initial velocity loss
        # disrupting the shared encoder.
        pred_vel = (torch.tanh(self.head_vel(h)) * self.max_speed).view(
            b, self.horizon, 2)
        uncertainty = torch.sigmoid(self.head_unc(h)).view(b, self.horizon, 2)
        return pred_traj, pred_vel, uncertainty

    def loss(self, pred_traj, pred_vel, target_traj, target_vel):
        return (F.mse_loss(pred_traj, target_traj)
                + F.mse_loss(pred_vel, target_vel))

    @torch.no_grad()
    def predict_single(self, history, n_steps=None):
        """Predict future positions for one agent history (seq_len, dim).

        Returns a numpy array (n_steps, 2).
        """
        n_steps = n_steps or self.horizon
        pred_traj, _, _ = self.forward(history)
        n = min(n_steps, self.horizon)
        return pred_traj[0, :n].cpu().numpy()


# ---------------------------------------------------------------------- #
# MultiAgentPredictor
# ---------------------------------------------------------------------- #
class MultiAgentPredictor:
    """Tracks several agents and predicts all of their futures at once."""

    def __init__(self, state_dim, max_agents=10, horizon=8, hidden_dim=128,
                 seq_len=5, device="cpu"):
        self.state_dim = state_dim
        self.max_agents = max_agents
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.predictor = TrafficPredictor(
            state_dim, horizon=horizon, hidden_dim=hidden_dim, device=device)
        self.histories = {}        # agent_id -> list of [x, y, vx, vy, class]
        self.agent_types = {}      # agent_id -> int class

    def add_agent(self, agent_id, agent_type):
        if agent_id not in self.histories:
            self.histories[agent_id] = []
        self.agent_types[agent_id] = int(agent_type)

    def observe_agent(self, agent_id, position, velocity):
        if agent_id not in self.histories:
            self.add_agent(agent_id, 0)
        cls = self.agent_types.get(agent_id, 0)
        self.histories[agent_id].append(
            [float(position[0]), float(position[1]),
             float(velocity[0]), float(velocity[1]), float(cls)]
        )
        # Keep only the most recent seq_len observations.
        if len(self.histories[agent_id]) > self.seq_len:
            self.histories[agent_id] = self.histories[agent_id][-self.seq_len:]

    def _padded(self, hist):
        """Left-pad a short history to seq_len by repeating its first row."""
        hist = list(hist)
        if not hist:
            return np.zeros((self.seq_len, 5), dtype=np.float32)
        while len(hist) < self.seq_len:
            hist.insert(0, hist[0])
        return np.asarray(hist[-self.seq_len:], dtype=np.float32)

    @torch.no_grad()
    def predict_all(self):
        """Returns dict agent_id -> (trajectory, velocity, uncertainty)."""
        ids = [a for a, h in self.histories.items() if len(h) > 0]
        if not ids:
            return {}
        batch = np.stack([self._padded(self.histories[a]) for a in ids], axis=0)
        traj, vel, unc = self.predictor(batch)
        traj, vel, unc = (traj.cpu().numpy(), vel.cpu().numpy(), unc.cpu().numpy())
        return {a: (traj[i], vel[i], unc[i]) for i, a in enumerate(ids)}

    def get_collision_risk(self, ego_pos, ego_traj, agent_predictions):
        """Collision risk between the ego trajectory and predicted agents."""
        traj = ego_traj if ego_traj is not None else [ego_pos]
        return compute_collision_risk(traj, agent_predictions)
