"""All hyperparameters for the Dreamer-PPO CARLA project."""
from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    # 48 = ego(6) + lane(4) + traffic(3) + 5 vehicle blocks x 5
    #      (ahead/behind/left/right/nearest) + 2 VRUs x 5.
    # (The Tier-1 spec said "42"; that is an arithmetic slip — 28 + 4 new
    #  5-dim vehicle blocks = 48, which the VRU index constants 38/43 require.)
    state_dim: int = 48
    action_dim: int = 4
    host: str = "localhost"
    port: int = 2000
    town: str = "Town01"
    fps: int = 10
    max_episode_steps: int = 1000

    # World model
    wm_hidden: int = 256
    lr_wm: float = 3e-4
    wm_batch_size: int = 256
    wm_warmup_steps: int = 1000   # transitions trained before dreaming starts

    # PPO
    hidden: int = 256
    lr_policy: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_size: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    num_episodes: int = 1000

    # Dreaming
    dream_k: int = 5
    w_progress: float = 1.0
    w_risk: float = 2.0
    w_value: float = 0.5

    # Reward weights
    w_prog: float = 1.0
    w_vru: float = 2.0
    w_safe: float = 1.0
    w_comfort: float = 0.1
    w_rules: float = 0.5

    # General vehicle safety (Tier 1: rear/side traffic awareness)
    w_vehicle: float = 1.0              # vehicle-safety weight (VRU stays 2.0)
    rear_risk_threshold: float = 3.0    # seconds TTC for a rear-collision penalty
    vehicle_proximity_sigma: float = 5.0    # distance scale for proximity penalty
    min_lane_change_clearance: float = 2.0  # min safe side gap for a lane change

    # Map-agnostic features (Tier 2: generalization to unknown towns)
    use_map_agnostic_features: bool = True   # augment state with computed features
    # 55 = 48 base + 7 features (in_lane_center + road-type one-hot[4] +
    # visibility + oncoming). The brief said "49"; that assumed the 42-dim
    # pre-Tier-1 state — with the 48-dim base it is 55.
    augmented_state_dim: int = 55

    # Defensive driving mode (for unknown maps)
    defensive_mode: bool = False             # start in defensive mode
    unknown_map_detection: str = "manual"    # 'manual' or 'gps_comparison'
    defensive_w_vru_mult: float = 1.5        # increase VRU weight
    defensive_w_vehicle_mult: float = 1.5    # increase vehicle-safety weight
    defensive_dream_k: int = 8               # more candidate actions
    defensive_horizon_max: int = 5           # deeper lookahead
    defensive_disable_risky_maneuvers: bool = True
    sigma_d: float = 5.0
    lambda_ttc: float = 0.5
    tau_ttc: float = 2.0
    lambda_cross: float = 1.0
    eta1: float = 0.5  # lane departure weight in safety term
    eta2: float = 0.3  # general risk weight in safety term

    # S-DBS incremental-testing overrides.
    # When sdbs_force_fixed_params is True, SDBSPlanner.get_search_params
    # ignores difficulty-based auto-scaling and returns the fixed values below.
    # Start any new integration at horizon=1, groups=1 (equivalent to greedy
    # one-step dreaming), confirm it matches the plain dreamer variant, then
    # increase the horizon gradually to isolate where behaviour degrades.
    sdbs_force_fixed_params: bool = False
    sdbs_fixed_horizon: int = 1
    sdbs_fixed_groups: int = 1
    sdbs_fixed_beam_width: int = 4
