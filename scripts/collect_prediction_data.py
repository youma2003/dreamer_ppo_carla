"""Pre-collect agent trajectory data for the traffic predictor.

Runs a number of episodes, extracts every VRU/vehicle history->future window,
and saves them to a pickle file so training can load pre-collected data instead
of collecting online.

Usage:
    python scripts/collect_prediction_data.py --episodes 500 \
        --save-path data/trajectories.pkl [--mock]
"""
import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import Config
from env.carla_env import CarlaEnv
from models.traffic_predictor import TrafficPredictor
from training.traffic_prediction_trainer import TrafficPredictionTrainer


def collect(episodes=500, save_path="data/trajectories.pkl", mock=False,
            max_steps=200):
    config = Config()
    env = CarlaEnv(mock=mock, config=config)
    predictor = TrafficPredictor(config.state_dim,
                                 horizon=config.predict_horizon,
                                 hidden_dim=config.tp_hidden_dim)
    trainer = TrafficPredictionTrainer(predictor, config)

    print(f"Collecting {episodes} episodes "
          f"({'mock' if mock else 'CARLA'} mode)...")
    try:
        trainer.collect_trajectories(env, num_episodes=episodes,
                                     max_steps=max_steps)
    finally:
        env.close()

    buf = trainer.trajectory_buffer
    data = list(zip(buf.histories, buf.futures, buf.types))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} trajectories to {save_path}")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory data")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save-path", default="data/trajectories.pkl")
    parser.add_argument("--mock", action="store_true",
                        help="run without CARLA installed")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    collect(args.episodes, args.save_path, mock=args.mock,
            max_steps=args.max_steps)


if __name__ == "__main__":
    main()
