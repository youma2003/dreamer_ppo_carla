"""Traffic / pedestrian prediction validation — runs with NO CARLA installed.

Run with:  python tests/test_traffic_prediction.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from configs.config import Config
from configs.sdbs_config import SDBSConfig
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from models.traffic_predictor import (
    TrafficPredictor, MultiAgentPredictor, compute_collision_risk,
)
from utils.trajectory_utils import TrajectoryBuffer
from training.traffic_prediction_trainer import TrafficPredictionTrainer
from planning.sdbs_planner import SDBSPlanner


def ok(name, result=""):
    print(f"✅ {name} : {result}")


# 1 ------------------------------------------------------------------- #
def test_predictor_shapes():
    pred = TrafficPredictor(state_dim=28, horizon=8, hidden_dim=64)
    x = torch.randn(8, 5, 5)
    traj, vel, unc = pred(x)
    assert traj.shape == (8, 8, 2), traj.shape
    assert vel.shape == (8, 8, 2)
    assert unc.shape == (8, 8, 2)
    assert torch.all((unc >= 0) & (unc <= 1))
    ok("traffic_predictor_shapes", "output dimensions correct")


# 2 ------------------------------------------------------------------- #
def test_trajectory_buffer():
    buf = TrajectoryBuffer(capacity=1000)
    for i in range(500):
        history = np.zeros((5, 5), dtype=np.float32)
        history[0, 0] = float(i)               # unique marker per trajectory
        future = np.ones((8, 5), dtype=np.float32) * i
        buf.add_trajectory(None, history, future, agent_type=i % 3)
    assert len(buf) == 500
    h, f, t = buf.get_batch(32)
    assert h.shape == (32, 5, 5) and f.shape == (32, 8, 5) and t.shape == (32,)
    markers = h[:, 0, 0].numpy()
    assert len(set(markers.tolist())) == 32     # sampled without replacement
    ok("trajectory_buffer", "storage and sampling work")


# 3 ------------------------------------------------------------------- #
def test_prediction_learning():
    cfg = Config()
    cfg.lr_tp = 1e-2
    pred = TrafficPredictor(state_dim=28, horizon=8, hidden_dim=64)
    trainer = TrafficPredictionTrainer(pred, cfg)
    seq_len, horizon = 5, 8
    rng = np.random.default_rng(0)
    for _ in range(256):
        x0, y0 = rng.uniform(-5, 5, size=2)
        vx, vy = rng.uniform(0, 0.6, size=2)
        steps = np.arange(seq_len + horizon)
        xs = x0 + steps * vx
        ys = y0 + steps * vy
        series = np.stack([xs, ys,
                           np.full_like(xs, vx), np.full_like(ys, vy),
                           np.zeros_like(xs)], axis=-1).astype(np.float32)
        trainer.trajectory_buffer.add_trajectory(
            None, series[:seq_len], series[seq_len:], 0)

    ade = None
    for _ in range(100):
        ade = trainer.update(trainer.trajectory_buffer.get_batch(64))["ade"]
    assert ade < 0.5, ade
    ok("prediction_learning", f"synthetic patterns learned, ADE={ade:.3f}")


# 4 ------------------------------------------------------------------- #
def test_multi_agent_predictor():
    mp = MultiAgentPredictor(state_dim=28, horizon=8, hidden_dim=64, seq_len=5)
    for a in range(5):
        mp.add_agent(f"agent_{a}", agent_type=a % 3)
        for t in range(5):
            mp.observe_agent(f"agent_{a}", (t + a, t), (1.0, 0.5))
    preds = mp.predict_all()
    assert len(preds) == 5
    for traj, vel, unc in preds.values():
        assert traj.shape == (8, 2) and vel.shape == (8, 2) and unc.shape == (8, 2)
    ok("multi_agent_predictor", "all agents predicted")


# 5 ------------------------------------------------------------------- #
def test_collision_risk():
    ego_traj = np.array([[t, 0.0] for t in range(8)], dtype=np.float32)
    # VRU sits far away except at t=5 where it lands exactly on the ego path.
    vru = np.full((8, 2), 50.0, dtype=np.float32)
    vru[5] = [5.0, 0.0]
    risk = compute_collision_risk(ego_traj, {"vru": vru})
    assert risk > 0.5, risk
    ok("collision_risk", "detected crossing trajectory")


# 6 ------------------------------------------------------------------- #
def test_tp_trainer_update():
    pred = TrafficPredictor(state_dim=28, horizon=8, hidden_dim=64)
    trainer = TrafficPredictionTrainer(pred, Config())
    rng = np.random.default_rng(1)
    histories = rng.standard_normal((32, 5, 5)).astype(np.float32)
    futures = rng.standard_normal((32, 8, 5)).astype(np.float32) * 2.0
    types = np.zeros(32, dtype=np.int64)
    batch = (histories, futures, types)
    first = trainer.update(batch)["loss"]
    for _ in range(20):
        last = trainer.update(batch)["loss"]
    assert last < first, (first, last)
    ok("tp_trainer_update", f"loss={last:.4f} after training")


# 7 ------------------------------------------------------------------- #
def test_carla_env_tracking():
    env = CarlaEnv(mock=True, config=Config())
    env.reset()
    for _ in range(5):
        env.step(np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32))
    hist = env.get_agent_histories()
    assert "vru0" in hist and "vru1" in hist and "veh0" in hist
    for arr in hist.values():
        assert arr.shape == (env.history_length, 5)
    env.close()
    ok("carla_env_tracking", "agent histories recorded")


# 8 ------------------------------------------------------------------- #
def test_full_prediction_pipeline():
    cfg = SDBSConfig()
    cfg.compute_budget = 40
    policy = ActorCritic(cfg.state_dim, cfg.action_dim, cfg.hidden)
    wm = WorldModel(cfg.state_dim, cfg.action_dim, cfg.wm_hidden)
    predictor = TrafficPredictor(cfg.state_dim, horizon=cfg.predict_horizon,
                                 hidden_dim=64)
    planner = SDBSPlanner(policy, wm, policy, cfg,
                          traffic_predictor=predictor)
    env = CarlaEnv(mock=True, config=cfg)
    used_predictions = False
    for _ in range(3):
        obs = env.reset()
        for _ in range(6):           # warm up agent histories
            obs, _r, done, info = env.step(
                np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32))
            if done:
                obs = env.reset()
        info["agent_histories"] = env.get_agent_histories()
        _action, _plan, meta = planner.plan(obs, info)
        assert "collision_risk" in meta
        used_predictions = used_predictions or meta["predicted_agents"] > 0
    env.close()
    assert used_predictions
    ok("full_prediction_pipeline", "planning uses predictions")


def main():
    print("Running Traffic Prediction tests (no CARLA needed)...\n")
    torch.manual_seed(0)
    np.random.seed(0)
    test_predictor_shapes()
    test_trajectory_buffer()
    test_prediction_learning()
    test_multi_agent_predictor()
    test_collision_risk()
    test_tp_trainer_update()
    test_carla_env_tracking()
    test_full_prediction_pipeline()
    print("\n✅ ALL TRAFFIC PREDICTION TESTS PASSED")


if __name__ == "__main__":
    main()
