"""Quick, visible demo of the learnable components on mock/synthetic data.

Mock CARLA states are random noise, so RL returns never trend. But the
*learnable* pieces do show real curves, which this script makes visible:

  1. predictor_learning.png  - traffic predictor ADE/FDE dropping over training
  2. prediction_example.png  - a predicted trajectory overlaid on ground truth
  3. collision_risk.png       - collision risk spiking when a VRU crosses the ego
  4. world_model_loss.png     - world-model loss decreasing on structured data

Run:  python scripts/demo_mock_results.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import Config
from models.traffic_predictor import TrafficPredictor, compute_collision_risk
from models.world_model import WorldModel
from training.traffic_prediction_trainer import TrafficPredictionTrainer
from training.world_model_trainer import WorldModelTrainer

OUT = os.path.join(ROOT, "demo_plots")
os.makedirs(OUT, exist_ok=True)
SEQ, HORIZON = 5, 8


# --------------------------------------------------------------------- #
def make_linear_dataset(n, rng):
    """Linear-motion agents: constant velocity, the predictor can learn this."""
    data = []
    for _ in range(n):
        x0, y0 = rng.uniform(-5, 5, size=2)
        vx, vy = rng.uniform(-0.6, 0.6, size=2)
        steps = np.arange(SEQ + HORIZON)
        xs, ys = x0 + steps * vx, y0 + steps * vy
        series = np.stack([xs, ys, np.full_like(xs, vx),
                           np.full_like(ys, vy), np.zeros_like(xs)], -1).astype(np.float32)
        data.append((series[:SEQ], series[SEQ:]))
    return data


def demo_predictor():
    print("[1/4] Training traffic predictor on synthetic linear motion...")
    rng = np.random.default_rng(0)
    cfg = Config()
    cfg.lr_tp = 1e-2
    pred = TrafficPredictor(cfg.state_dim, horizon=HORIZON, hidden_dim=64)
    trainer = TrafficPredictionTrainer(pred, cfg)
    for hist, fut in make_linear_dataset(512, rng):
        trainer.trajectory_buffer.add_trajectory(None, hist, fut, 0)

    ades, fdes = [], []
    for _ in range(200):
        stats = trainer.update(trainer.trajectory_buffer.get_batch(64))
        ades.append(stats["ade"])
        fdes.append(stats["fde"])
    print(f"      ADE: {ades[0]:.2f} -> {ades[-1]:.2f} m   "
          f"FDE: {fdes[0]:.2f} -> {fdes[-1]:.2f} m")

    plt.figure(figsize=(8, 4.5))
    plt.plot(ades, label="ADE (avg displacement error)")
    plt.plot(fdes, label="FDE (final displacement error)")
    plt.xlabel("Training step")
    plt.ylabel("Error (m)")
    plt.title("Traffic predictor learning on synthetic trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "predictor_learning.png"), dpi=120)
    plt.close()

    # One predicted trajectory vs its ground truth.
    hist, fut = make_linear_dataset(1, np.random.default_rng(7))[0]
    pred_traj = pred.predict_single(hist, n_steps=HORIZON)
    plt.figure(figsize=(6, 6))
    plt.plot(hist[:, 0], hist[:, 1], "ko-", label="history")
    plt.plot(fut[:, 0], fut[:, 1], "g^-", label="ground-truth future")
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], "rx--", label="predicted future")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Predicted vs ground-truth trajectory (after training)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prediction_example.png"), dpi=120)
    plt.close()


def demo_collision_risk():
    print("[2/4] Collision-risk demo (a VRU crossing the ego path)...")
    T = 10
    ego = np.stack([np.arange(T), np.zeros(T)], axis=-1).astype(np.float32)
    # VRU walks down toward the ego lane, crossing x=5 exactly at t=5.
    vru = np.stack([np.full(T, 5.0), np.linspace(6, -4, T)], axis=-1).astype(np.float32)

    sigma = 1.5
    dist = np.linalg.norm(ego - vru, axis=-1)
    prox = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    risk = compute_collision_risk(ego, {"vru": vru}, sigma=sigma)
    print(f"      min distance = {dist.min():.2f} m at t={int(dist.argmin())}, "
          f"collision risk = {risk:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ego[:, 0], ego[:, 1], "bo-", label="ego")
    ax1.plot(vru[:, 0], vru[:, 1], "rs-", label="VRU (predicted)")
    ax1.scatter([ego[5, 0]], [ego[5, 1]], s=200, facecolors="none",
                edgecolors="orange", linewidths=2, label="conflict point")
    ax1.set_title("Trajectories (top-down)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    ax2.plot(prox, "m^-", label="collision proximity")
    ax2.plot(dist / dist.max(), "c.--", label="distance (normalized)")
    ax2.axhline(risk, color="r", ls=":", label=f"risk = {risk:.2f}")
    ax2.set_title("Collision risk over the horizon")
    ax2.set_xlabel("timestep")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "collision_risk.png"), dpi=120)
    plt.close()


def demo_world_model():
    print("[3/4] World-model loss on a structured (learnable) batch...")
    cfg = Config()
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    trainer = WorldModelTrainer(wm, cfg)
    rng = torch.Generator().manual_seed(0)
    states = torch.randn(256, cfg.state_dim, generator=rng)
    actions = torch.randn(256, cfg.action_dim, generator=rng)
    # Deterministic targets so the model has a real function to fit.
    next_states = torch.tanh(states + 0.1)
    batch = {
        "states": states, "actions": actions, "next_states": next_states,
        "risk_targets": torch.sigmoid(states[:, :1].squeeze(-1)),
        "progress_targets": states[:, 12],
    }
    losses = [trainer.update(batch)["loss_wm"] for _ in range(200)]
    print(f"      WM loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Total loss")
    plt.title("World-model loss decreasing on structured data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "world_model_loss.png"), dpi=120)
    plt.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    torch.manual_seed(0)
    np.random.seed(0)
    demo_predictor()
    demo_collision_risk()
    demo_world_model()
    print("[4/4] Done.")
    print(f"\nPlots written to: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
