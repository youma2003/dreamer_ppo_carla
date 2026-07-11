"""CARLA environment wrapper with a fully functional mock mode.

In mock mode no `carla` import is attempted, so every component of the
project can be developed and tested locally without CARLA installed.
"""
import numpy as np

from configs.config import Config
from rewards.vru_reward import (
    compute_reward, compute_vru_risk_target, ROUTE_PROGRESS, LANE_OFFSET,
    EGO_X, EGO_Y, EGO_SPEED, EGO_HEADING,
    VEHICLE_AHEAD_DIST, VEHICLE_BEHIND_DIST, VEHICLE_LEFT_DIST,
    VEHICLE_RIGHT_DIST, VEHICLE_NEAREST_DIST,
    VRU1_DIST, VRU2_DIST,
)


# ---------------------------------------------------------------------- #
# State-vector contract (single source of truth)
# ---------------------------------------------------------------------- #
# ``STATE_DIM`` is the full augmented state consumed by the policy in the
# S-DBS path: 48 base env features + 7 map-agnostic features. The base env
# itself emits ``config.state_dim`` (48 by default); the map-agnostic wrapper
# appends the final 7. Update this table if the layout ever changes.
STATE_DIM = 55
STATE_LAYOUT = {
    'ego': (0, 6),
    'lane': (6, 10),
    'traffic': (10, 13),
    'vehicle_ahead': (13, 18),
    'vehicle_behind': (18, 23),
    'vehicle_left': (23, 28),
    'vehicle_right': (28, 33),
    'vehicle_nearest': (33, 38),
    'vru': (38, 48),
    'map_agnostic': (48, 55),
}


def validate_state_vector(state, expected_dim=STATE_DIM):
    """Raise immediately if the state shape doesn't match ``expected_dim``.

    Never silently pad, truncate, or reshape — a wrong dimension almost always
    means the state-building code and the trained checkpoint disagree, which
    silently produces garbage. ``expected_dim`` defaults to the full augmented
    contract (``STATE_DIM``); callers that operate on the base env state pass
    their own configured dimension so the check fires on true mismatches
    without conflating the base (48) and augmented (55) layouts.
    """
    dim = state.shape[-1]
    if dim != expected_dim:
        raise ValueError(
            f"State vector has {dim} dims, expected {expected_dim}. "
            f"Layout (full augmented state): {STATE_LAYOUT}. "
            f"This usually means the state-building code and the trained "
            f"checkpoint disagree on dimensions."
        )


def classify_vehicles(ego_x, ego_y, ego_heading, lane_width, vehicles,
                      default_dist=100.0):
    """Sort surrounding vehicles into ahead/behind/left/right/nearest blocks.

    ``vehicles`` is a list of dicts with keys ``x, y, speed, heading`` in world
    coordinates. Each output block is ``[dist, speed, heading, local_x,
    local_y]`` in the ego frame (local_x>0 ahead, local_y>0 to the left).
    Missing directions default to a far, stationary placeholder.
    """
    default = [default_dist, 0.0, 0.0, 0.0, 0.0]
    ahead = behind = left = right = nearest = None
    min_dist = float("inf")
    cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
    half_lane = max(0.5, float(lane_width) / 2.0)

    for v in vehicles:
        rel_x, rel_y = float(v["x"]) - ego_x, float(v["y"]) - ego_y
        local_x = rel_x * cos_h - rel_y * sin_h
        local_y = rel_x * sin_h + rel_y * cos_h
        dist = float(np.hypot(local_x, local_y))
        block = [dist, float(v.get("speed", 0.0)), float(v.get("heading", 0.0)),
                 local_x, local_y]

        if dist < min_dist:
            min_dist, nearest = dist, block
        if local_x > 0 and (ahead is None or dist < ahead[0]):
            ahead = block
        elif local_x < 0 and (behind is None or dist < behind[0]):
            behind = block
        if local_y > half_lane and (left is None or dist < left[0]):
            left = block
        elif local_y < -half_lane and (right is None or dist < right[0]):
            right = block

    return {
        "ahead": ahead or default, "behind": behind or default,
        "left": left or default, "right": right or default,
        "nearest": nearest or default,
    }


class CarlaEnv:
    """Flat-vector CARLA driving environment.

    State vector (dim=28):
      ego (6):     x, y, speed, heading, acc_x, acc_y
      lane (4):    lane_offset, lane_width, road_curvature, is_junction
      traffic (3): traffic_light_state, dist_to_light, route_progress
      vehicles (5):nearest_vehicle dist, speed, heading, rel_x, rel_y
      VRU (10):    up to 2 VRUs, each: dist, speed, heading, rel_x, rel_y

    Action (Box(4,)): [steering(-1,1), throttle(0,1), brake(0,1), stop_continue(0,1)]
    """

    def __init__(self, mock=False, config=None):
        self.config = config or Config()
        self.mock = mock
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.rng = np.random.default_rng()

        self._step_count = 0
        self._state = None
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._scenario_id = None        # set by reset_to_scenario (S-DBS curriculum)

        # CARLA handles (real mode only).
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self._collision_flag = False

        if not self.mock:
            self._init_carla()

    # ------------------------------------------------------------------ #
    # Real CARLA initialization
    # ------------------------------------------------------------------ #
    def _init_carla(self):
        try:
            import carla  # imported only in real mode
        except Exception as exc:  # pragma: no cover - requires CARLA
            raise RuntimeError(
                "Could not import `carla`. Run with mock=True to develop "
                "without a CARLA installation."
            ) from exc

        self._carla = carla
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.config.town)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.fps
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()

    def _spawn_actors(self):  # pragma: no cover - requires CARLA
        carla = self._carla
        self._destroy_actors()

        bp = self.blueprint_library.filter("vehicle.*")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        transform = self.rng.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(bp, transform)

        # Camera sensor.
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)

        # Collision sensor.
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self._collision_flag = False
        self.collision_sensor.listen(lambda e: setattr(self, "_collision_flag", True))

    def _destroy_actors(self):  # pragma: no cover - requires CARLA
        for actor in (self.camera, self.collision_sensor, self.vehicle):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.camera = None
        self.collision_sensor = None
        self.vehicle = None

    # ------------------------------------------------------------------ #
    # Mock helpers
    # ------------------------------------------------------------------ #
    def _random_state(self):
        s = np.zeros(self.state_dim, dtype=np.float32)
        # ego
        s[0:2] = self.rng.uniform(-100, 100, size=2)       # x, y
        s[2] = self.rng.uniform(0, 15)                      # speed
        s[3] = self.rng.uniform(-np.pi, np.pi)              # heading
        s[4:6] = self.rng.uniform(-3, 3, size=2)            # acc
        # lane
        s[6] = self.rng.uniform(-1.5, 1.5)                  # lane_offset
        s[7] = self.rng.uniform(3.0, 4.0)                   # lane_width
        s[8] = self.rng.uniform(-0.1, 0.1)                  # curvature
        s[9] = float(self.rng.integers(0, 2))               # is_junction
        # traffic
        s[10] = float(self.rng.integers(0, 3))              # light state 0/1/2
        s[11] = self.rng.uniform(0, 50)                     # dist_to_light
        s[12] = self.rng.uniform(0, 1)                      # route_progress
        # vehicle blocks: ahead[13], behind[18], left[23], right[28], nearest[33]
        for base in (VEHICLE_AHEAD_DIST, VEHICLE_BEHIND_DIST, VEHICLE_LEFT_DIST,
                     VEHICLE_RIGHT_DIST, VEHICLE_NEAREST_DIST):
            s[base] = self.rng.uniform(3, 60)               # dist
            s[base + 1] = self.rng.uniform(0, 15)           # speed
            s[base + 2] = self.rng.uniform(-np.pi, np.pi)   # heading
            s[base + 3:base + 5] = self.rng.uniform(-30, 30, size=2)  # rel_x, rel_y
        # VRUs at [38..42] and [43..47]
        for base in (VRU1_DIST, VRU2_DIST):
            s[base] = self.rng.uniform(1, 40)               # dist
            s[base + 1] = self.rng.uniform(0, 3)            # speed
            s[base + 2] = self.rng.uniform(-np.pi, np.pi)   # heading
            s[base + 3:base + 5] = self.rng.uniform(-20, 20, size=2)
        return s

    def _mock_info(self, state, action, progress):
        """Build a realistic info dict with mock VRU + rule signals."""
        collision = bool(self.rng.random() < 0.05)        # 5% per step
        lane_departure = bool(self.rng.random() < 0.10)   # 10% per step
        info = {
            "action": np.asarray(action, dtype=np.float32),
            "prev_action": self._prev_action.copy(),
            "collision": collision,
            "lane_departure": lane_departure,
            "collision_with_vehicle": bool(self.rng.random() < 0.04),  # 4%
            "red_light_violation": bool(self.rng.random() < 0.03),   # 3%
            "stop_sign_violation": bool(self.rng.random() < 0.02),   # 2%
            "crosswalk_conflict": bool(self.rng.random() < 0.05),    # 5%
            "general_risk": float(self.rng.uniform(0.0, 0.3)),
            "weather": "clear",
            "time_of_day": "day",
            # `progress` is the real per-step route advance (used as the
            # world-model progress target); it stays consistent with the
            # reward instead of being pure noise.
            "progress": float(progress),
            "route_completion": float(state[ROUTE_PROGRESS]),
            "vru_collisions": int(collision),
            "lane_departures": int(lane_departure),
        }
        info["vru_risk"] = compute_vru_risk_target(state, info, self.config)
        info.update(self._compute_safety_lists(state))
        return info

    # ------------------------------------------------------------------ #
    # Per-step safety signals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_safety_lists(state):
        """Per-step TTC / distance lists for VRUs and vehicles (for tracking).

        Returns a dict with parallel ``*_ttc_list`` / ``*_distance_list`` and,
        for vehicles, a ``vehicle_directions`` list (front/rear/left/right).
        TTC uses the direction-appropriate closing speed; non-closing pairs get
        a large TTC so they are not counted as near-misses.
        """
        s = np.asarray(state, dtype=np.float32)
        ego_speed = float(s[EGO_SPEED])

        vru_ttc, vru_dist = [], []
        for idx in (VRU1_DIST, VRU2_DIST):
            dist = float(s[idx])
            if dist <= 0:
                continue
            vru_dist.append(dist)
            vru_ttc.append(dist / max(0.1, ego_speed))

        veh_ttc, veh_dist, veh_dirs = [], [], []
        for base, direction in ((VEHICLE_AHEAD_DIST, "front"),
                                (VEHICLE_BEHIND_DIST, "rear"),
                                (VEHICLE_LEFT_DIST, "left"),
                                (VEHICLE_RIGHT_DIST, "right")):
            dist = float(s[base])
            if dist <= 0 or dist >= 100:        # skip placeholder / far vehicles
                continue
            vspeed = float(s[base + 1])
            if direction == "front":
                closing = ego_speed - vspeed
            elif direction == "rear":
                closing = vspeed - ego_speed
            else:
                closing = abs(ego_speed - vspeed)
            ttc = dist / closing if closing > 0 else 999.0
            veh_dist.append(dist)
            veh_ttc.append(ttc)
            veh_dirs.append(direction)

        return {
            "vru_ttc_list": vru_ttc, "vru_distance_list": vru_dist,
            "vehicle_ttc_list": veh_ttc, "vehicle_distance_list": veh_dist,
            "vehicle_directions": veh_dirs,
        }

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self):
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.mock:
            self._state = self._random_state()
            validate_state_vector(self._state, self.state_dim)
            return self._state.copy()

        # Real mode.  # pragma: no cover - requires CARLA
        self._spawn_actors()
        for _ in range(self.config.fps):
            self.world.tick()
        self._state = self._observe()
        validate_state_vector(self._state, self.state_dim)
        return self._state.copy()

    def reset_to_scenario(self, scenario_id):
        """Reset the episode under a named curriculum scenario.

        In mock mode this just records the scenario id and resets normally; in
        real mode a scenario would select the spawn/traffic configuration. The
        S-DBS curriculum uses the id to track per-scenario success and priority.
        """
        self._scenario_id = scenario_id
        return self.reset()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.mock:
            next_state = self._random_state()
            # Advance route progress monotonically for realism.
            progress = float(self.rng.uniform(0.0, 0.05))
            next_state[ROUTE_PROGRESS] = np.clip(
                self._state[ROUTE_PROGRESS] + progress, 0.0, 1.0
            )
            info = self._mock_info(next_state, action, progress)
            reward, components = compute_reward(
                self._state, next_state, action, self._prev_action,
                info, self.config,
            )
            info["reward_components"] = components
            self._step_count += 1
            done = bool(
                info["collision"]
                or next_state[ROUTE_PROGRESS] >= 1.0
                or self._step_count >= self.config.max_episode_steps
            )
            self._prev_action = action.copy()
            self._state = next_state
            validate_state_vector(next_state, self.state_dim)
            return next_state.copy(), reward, done, info

        # Real mode.  # pragma: no cover - requires CARLA
        self._apply_control(action)
        self.world.tick()
        next_state = self._observe()
        info = self._real_info(next_state, action)
        reward, components = compute_reward(
            self._state, next_state, action, self._prev_action,
            info, self.config,
        )
        info["reward_components"] = components
        self._step_count += 1
        done = bool(
            self._collision_flag
            or next_state[ROUTE_PROGRESS] >= 1.0
            or self._step_count >= self.config.max_episode_steps
        )
        self._prev_action = action.copy()
        self._state = next_state
        validate_state_vector(next_state, self.state_dim)
        return next_state.copy(), reward, done, info

    # ------------------------------------------------------------------ #
    # Real-mode observation/control (requires CARLA)
    # ------------------------------------------------------------------ #
    def _apply_control(self, action):  # pragma: no cover - requires CARLA
        carla = self._carla
        steer = float(np.clip(action[0], -1, 1))
        throttle = float(np.clip(action[1], 0, 1))
        brake = float(np.clip(action[2], 0, 1))
        # stop_continue near 0 means "stop": override with full brake.
        if action[3] < 0.5:
            throttle, brake = 0.0, 1.0
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    def _observe(self):  # pragma: no cover - requires CARLA
        s = np.zeros(self.state_dim, dtype=np.float32)
        tf = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        acc = self.vehicle.get_acceleration()
        s[0] = tf.location.x
        s[1] = tf.location.y
        s[2] = float(np.linalg.norm([vel.x, vel.y, vel.z]))
        s[3] = np.deg2rad(tf.rotation.yaw)
        s[4] = acc.x
        s[5] = acc.y
        # Remaining fields would be filled from waypoints / actor lists.
        s[7] = 3.5  # nominal lane width

        # Tier-1: sort surrounding vehicles into ahead/behind/left/right/nearest.
        vehicles = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            loc = actor.get_location()
            avel = actor.get_velocity()
            vehicles.append({
                "x": loc.x, "y": loc.y,
                "speed": float(np.linalg.norm([avel.x, avel.y, avel.z])),
                "heading": np.deg2rad(actor.get_transform().rotation.yaw),
            })
        blocks = classify_vehicles(s[EGO_X], s[EGO_Y], s[EGO_HEADING], s[7],
                                   vehicles)
        for base, key in ((VEHICLE_AHEAD_DIST, "ahead"),
                          (VEHICLE_BEHIND_DIST, "behind"),
                          (VEHICLE_LEFT_DIST, "left"),
                          (VEHICLE_RIGHT_DIST, "right"),
                          (VEHICLE_NEAREST_DIST, "nearest")):
            s[base:base + 5] = blocks[key]
        return s

    def _real_info(self, state, action):  # pragma: no cover - requires CARLA
        collision = bool(self._collision_flag)
        lane_departure = bool(abs(state[LANE_OFFSET]) > 1.4)
        info = {
            "action": np.asarray(action, dtype=np.float32),
            "prev_action": self._prev_action.copy(),
            "collision": collision,
            "lane_departure": lane_departure,
            "collision_with_vehicle": collision,
            "red_light_violation": False,
            "stop_sign_violation": False,
            "crosswalk_conflict": False,
            "general_risk": 0.0,
            "progress": float(state[ROUTE_PROGRESS] - self._state[ROUTE_PROGRESS]),
            "route_completion": float(state[ROUTE_PROGRESS]),
            "vru_collisions": int(collision),
            "lane_departures": int(lane_departure),
        }
        info["vru_risk"] = compute_vru_risk_target(state, info, self.config)
        info.update(self._compute_safety_lists(state))
        return info

    def close(self):
        if not self.mock:  # pragma: no cover - requires CARLA
            try:
                self._destroy_actors()
                if self.world is not None:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    self.world.apply_settings(settings)
            except Exception:
                pass
