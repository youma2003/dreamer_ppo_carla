# Dreamer-PPO for CARLA Autonomous Driving

A compact **Dreamer-style world model + PPO** agent for urban autonomous
driving in [CARLA](https://carla.org/), with a strong focus on **VRU
(Vulnerable Road User) safety**.

The whole project runs in **mock mode without CARLA installed**, so it can be
developed and tested locally. The CARLA-specific code is only imported when
running for real (e.g. on the machine that hosts the simulator).

```
dreamer_ppo_carla/
├── env/
│   └── carla_env.py          # CARLA wrapper + fully functional mock mode
├── models/
│   ├── world_model.py        # MLP world model (next_state / risk / progress)
│   ├── actor_critic.py       # PPO policy (continuous Box(4) action)
│   └── rssm.py               # RSSM world model (upgrade path)
├── training/
│   ├── ppo.py                # PPO clipped loss + world-model update
│   ├── rollout_buffer.py     # GAE rollout buffer
│   └── dreamer_ppo.py        # main training loop + dreaming action selection
├── rewards/
│   └── vru_reward.py         # VRU safety reward terms
├── configs/
│   └── config.py             # all hyperparameters
├── tests/
│   └── test_mock.py          # full pipeline test, no CARLA needed
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

`numpy` and `torch` are all you need for mock mode. `carla` is optional and
only required for real runs (install a wheel matching your CARLA server, e.g.
`pip install carla==0.9.15`).

## Quick start (no CARLA)

Run the full end-to-end test suite — this exercises every component:

```bash
python tests/test_mock.py
```

Expected final line:

```
✅ ALL TESTS PASSED
```

Run a short mock training session:

```bash
python -m training.dreamer_ppo --mock --episodes 5
```

## Run for real (on the CARLA host)

Start the CARLA server, then:

```bash
python -m training.dreamer_ppo --episodes 1000
```

(omit `--mock`). The environment connects to `localhost:2000`, loads `Town01`,
spawns an ego vehicle with camera + collision sensors, and steps in
synchronous mode at `fps` from the config.

## State & action spaces

**State** — flat `float32` vector, `dim = 28`:

| group     | size | fields |
|-----------|------|--------|
| ego       | 6    | x, y, speed, heading, acc_x, acc_y |
| lane      | 4    | lane_offset, lane_width, road_curvature, is_junction |
| traffic   | 3    | traffic_light_state, dist_to_light, route_progress |
| vehicles  | 5    | nearest dist, speed, heading, rel_x, rel_y |
| VRU       | 10   | up to 2 VRUs × (dist, speed, heading, rel_x, rel_y) |

**Action** — continuous `Box(4)`:
`[steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1], stop_continue ∈ [0,1]]`

## How it works

1. **World model** (`WorldModel`) predicts the next state, a VRU **risk**
   score in `[0,1]`, and **route progress** from `(state, action)`.
2. **Dreaming** (`select_action_with_dreaming`): sample `k` candidate actions,
   roll each through the world model, and pick the one with the best imagined
   score `w_progress·progress − w_risk·risk + w_value·V(next_state)`.
3. **PPO** (`update_ppo`) trains the `ActorCritic` policy on GAE advantages
   from the `RolloutBuffer`, while `update_world_model` regresses the world
   model against observed next states, risk targets, and progress targets.
4. **VRU reward** (`compute_reward`) combines progress, VRU proximity / TTC /
   crosswalk penalties, collision & lane-departure safety, comfort (action
   smoothness), and traffic-rule violations. All weights live in `config.py`.

## Upgrade path: RSSM

`models/rssm.py` provides a DreamerV3-style recurrent state-space model with
the same prediction heads (next_state / risk / progress). It can replace the
MLP `WorldModel` to enable latent imagination over multiple steps.

## Configuration

All hyperparameters are in `configs/config.py` as a single `@dataclass Config`
(environment, world model, PPO, dreaming, and reward weights). Edit there or
construct a `Config()` and override fields in code.
