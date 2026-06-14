"""VRU (Vulnerable Road User) safety reward terms.

State vector layout (dim=28):
  ego (6):     [0] x, [1] y, [2] speed, [3] heading, [4] acc_x, [5] acc_y
  lane (4):    [6] lane_offset, [7] lane_width, [8] road_curvature, [9] is_junction
  traffic (3): [10] traffic_light_state, [11] dist_to_light, [12] route_progress
  vehicles (5):[13] nearest_veh_dist, [14] speed, [15] heading, [16] rel_x, [17] rel_y
  VRU (10):    2 VRUs x (dist, speed, heading, rel_x, rel_y)
               VRU0: [18..22], VRU1: [23..27]
"""
import numpy as np

# Index constants for readability.
EGO_SPEED = 2
LANE_OFFSET = 6
LANE_WIDTH = 7
TL_STATE = 10
DIST_TO_LIGHT = 11
ROUTE_PROGRESS = 12
VRU_START = 18
VRU_STRIDE = 5
NUM_VRU = 2


def _vru_slice(state, i):
    """Return (dist, speed, heading, rel_x, rel_y) for VRU i."""
    base = VRU_START + i * VRU_STRIDE
    return state[base:base + VRU_STRIDE]


def compute_reward(state, next_state, info, config):
    """Compute the scalar reward for a transition.

    info may carry richer mock/real signals:
      collision (bool), lane_departure (bool), red_light (bool),
      stop_sign_violation (bool), crosswalk_conflict (float/bool),
      general_risk (float in [0,1]), progress (float), prev_action,
      action (current action).
    Falls back to values derived from the state vectors when absent.
    """
    state = np.asarray(state, dtype=np.float32)
    next_state = np.asarray(next_state, dtype=np.float32)

    # ---- progress: how much route_progress increased -------------------
    if "progress" in info:
        progress = float(info["progress"])
    else:
        progress = float(next_state[ROUTE_PROGRESS] - state[ROUTE_PROGRESS])

    # ---- vru_risk -------------------------------------------------------
    sigma_d = config.sigma_d
    vru_risk = 0.0
    for i in range(NUM_VRU):
        dist, speed, heading, rel_x, rel_y = _vru_slice(next_state, i)
        if dist <= 0:
            continue
        # proximity penalty
        vru_risk -= float(np.exp(-dist / sigma_d))
        # near-miss penalty using time-to-collision
        rel_speed = max(float(speed), 1e-3)
        ttc = dist / rel_speed
        if ttc < config.tau_ttc:
            vru_risk -= config.lambda_ttc
    # crosswalk blocking penalty
    crosswalk_conflict = float(info.get("crosswalk_conflict", 0.0))
    vru_risk -= config.lambda_cross * crosswalk_conflict

    # ---- safety ---------------------------------------------------------
    collision = float(info.get("collision", 0.0))
    lane_departure = float(info.get("lane_departure", 0.0))
    general_risk = float(info.get("general_risk", 0.0))
    safety = -collision - config.eta1 * lane_departure - config.eta2 * general_risk

    # ---- comfort: penalize action deltas --------------------------------
    action = np.asarray(info.get("action", np.zeros(config.action_dim)), dtype=np.float32)
    prev_action = np.asarray(
        info.get("prev_action", np.zeros(config.action_dim)), dtype=np.float32
    )
    delta = np.abs(action[:3] - prev_action[:3])  # steer, throttle, brake
    comfort = -float(np.sum(delta))

    # ---- rules ----------------------------------------------------------
    red_light = float(info.get("red_light", 0.0))
    stop_violation = float(info.get("stop_sign_violation", 0.0))
    rules = -red_light - stop_violation

    total = (
        config.w_prog * progress
        + config.w_vru * vru_risk
        + config.w_safe * safety
        + config.w_comfort * comfort
        + config.w_rules * rules
    )
    return float(total)


def compute_vru_risk_target(state, info, config):
    """Normalized VRU risk target in [0,1] for world-model supervision."""
    if "vru_risk" in info:
        return float(np.clip(info["vru_risk"], 0.0, 1.0))
    state = np.asarray(state, dtype=np.float32)
    risk = 0.0
    for i in range(NUM_VRU):
        dist = _vru_slice(state, i)[0]
        if dist > 0:
            risk += float(np.exp(-dist / config.sigma_d))
    return float(np.clip(risk / max(NUM_VRU, 1), 0.0, 1.0))
