"""Main Dreamer-PPO training loop with imagination-based action selection."""
import argparse

import numpy as np
import torch

from configs.config import Config
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from training.rollout_buffer import RolloutBuffer
from training.ppo import update_ppo, update_world_model
from training.logger import Logger
from rewards.vru_reward import ROUTE_PROGRESS


# ---------------------------------------------------------------------- #
# Dreaming action selection
# ---------------------------------------------------------------------- #
def select_action_with_dreaming(policy, world_model, state, k=5,
                                w_progress=1.0, w_risk=2.0, w_value=0.5):
    """Sample k candidate actions, roll each through the world model, and
    pick the one with the best imagined score.

    Returns (action, raw_action, log_prob, value) where `action` is the
    bounded action to send to the environment and `raw_action` is the
    pre-squash sample stored for the PPO update.
    """
    if not torch.is_tensor(state):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32))
    if state.dim() == 1:
        state = state.unsqueeze(0)

    best_score = -float("inf")
    best = None
    with torch.no_grad():
        for _ in range(k):
            action, log_prob, _value, raw_action = policy.act(state)
            next_state_hat, risk_hat, progress_hat = world_model(state, action)
            _, _, next_value = policy.forward(next_state_hat)
            score = (
                w_progress * progress_hat.squeeze(-1)
                - w_risk * risk_hat.squeeze(-1)
                + w_value * next_value
            )
            score = float(score.item())
            if score > best_score:
                best_score = score
                # value for the *current* state from the critic
                _, _, cur_value = policy.forward(state)
                best = (
                    action.squeeze(0),
                    raw_action.squeeze(0),
                    log_prob.squeeze(0),
                    cur_value.squeeze(0),
                )

    action, raw_action, log_prob, value = best
    return action.numpy(), raw_action.numpy(), float(log_prob.item()), float(value.item())


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train(config=None, mock=False, num_episodes=None, verbose=True,
          log_dir=None):
    config = config or Config()
    if num_episodes is not None:
        config.num_episodes = num_episodes

    env = CarlaEnv(mock=mock, config=config)
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden)
    world_model = WorldModel(config.state_dim, config.action_dim, config.wm_hidden)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)
    opt_wm = torch.optim.Adam(world_model.parameters(), lr=config.lr_wm)

    buffer = RolloutBuffer(
        config.rollout_size, config.state_dim, config.action_dim,
        gamma=config.gamma, lam=config.lam,
    )

    logger = Logger(log_dir) if log_dir else None
    history = []
    try:
        for episode in range(config.num_episodes):
            buffer.clear()
            obs = env.reset()
            ep_return = 0.0
            ep_collisions = 0
            ep_lane_departures = 0
            ep_components = {k: 0.0 for k in
                             ("progress", "vru_risk", "collision", "comfort", "rules")}
            route_completion = float(obs[ROUTE_PROGRESS])
            done = False

            # Fill the rollout buffer (may span several episodes).
            while not buffer.is_full():
                action, raw_action, log_prob, value = select_action_with_dreaming(
                    policy, world_model, obs,
                    k=config.dream_k, w_progress=config.w_progress,
                    w_risk=config.w_risk, w_value=config.w_value,
                )
                next_obs, reward, done, info = env.step(action)

                risk_target = float(info.get("vru_risk", 0.0))
                progress_target = float(info.get("progress", 0.0))

                buffer.store(obs, raw_action, reward, done, value, log_prob,
                             next_obs, risk_target, progress_target)
                ep_return += reward
                ep_collisions += int(bool(info.get("collision", False)))
                ep_lane_departures += int(bool(info.get("lane_departure", False)))
                comp = info.get("reward_components", {})
                for key in ep_components:
                    ep_components[key] += float(comp.get(key, 0.0))
                route_completion = float(next_obs[ROUTE_PROGRESS])
                obs = next_obs

                if done:
                    buffer.finish_path(last_value=0.0)
                    obs = env.reset()

            # Bootstrap the value of the final (unfinished) state.
            if buffer.path_start < buffer.ptr:
                with torch.no_grad():
                    _, _, last_v = policy.forward(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    )
                buffer.finish_path(last_value=float(last_v.item()))

            # ---- updates ---- #
            batch = buffer.get()
            n = batch["states"].shape[0]
            last_ppo, last_wm = {}, {}
            for _ in range(config.update_epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, config.batch_size):
                    mb_idx = idx[start:start + config.batch_size]
                    mb = {key: val[mb_idx] for key, val in batch.items()}
                    last_ppo = update_ppo(
                        policy, opt_pi, mb, clip_eps=config.clip_eps,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                        max_grad_norm=config.max_grad_norm,
                    )
                    last_wm = update_world_model(
                        world_model, opt_wm, mb, max_grad_norm=config.max_grad_norm
                    )

            record = {
                "episode": episode,
                "return": ep_return,
                "r_progress": ep_components["progress"],
                "r_vru": ep_components["vru_risk"],
                "r_collision": ep_components["collision"],
                "r_comfort": ep_components["comfort"],
                "r_rules": ep_components["rules"],
                "ppo_loss": last_ppo.get("loss", 0.0),
                "vf_loss": last_ppo.get("value_loss", 0.0),
                "entropy": last_ppo.get("entropy", 0.0),
                "wm_loss": last_wm.get("loss", 0.0),
                "vru_collisions": ep_collisions,
                "lane_departures": ep_lane_departures,
                "route_completion": route_completion,
                "collisions": ep_collisions,
            }
            history.append(record)
            if logger is not None:
                logger.log(episode, record)
            if verbose:
                print(
                    f"[ep {episode:4d}] return={ep_return:8.2f} "
                    f"ppo_loss={last_ppo.get('loss', 0.0):7.4f} "
                    f"wm_loss={last_wm.get('loss', 0.0):7.4f} "
                    f"collisions={ep_collisions}"
                )
    finally:
        env.close()
        if logger is not None:
            logger.close()

    return history


def main():
    parser = argparse.ArgumentParser(description="Dreamer-PPO for CARLA")
    parser.add_argument("--mock", action="store_true",
                        help="run without CARLA installed")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    train(config, mock=args.mock, num_episodes=args.episodes, log_dir="logs")


if __name__ == "__main__":
    main()
