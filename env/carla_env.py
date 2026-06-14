"""CARLA environment wrapper with a fully functional mock mode.

In mock mode no `carla` import is attempted, so every component of the
project can be developed and tested locally without CARLA installed.
"""
import numpy as np

from configs.config import Config
from rewards.vru_reward import compute_reward, compute_vru_risk_target, ROUTE_PROGRESS


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
        # nearest vehicle
        s[13] = self.rng.uniform(2, 50)                     # dist
        s[14] = self.rng.uniform(0, 15)                     # speed
        s[15] = self.rng.uniform(-np.pi, np.pi)             # heading
        s[16:18] = self.rng.uniform(-30, 30, size=2)        # rel_x, rel_y
        # VRUs
        for i in range(2):
            base = 18 + i * 5
            s[base] = self.rng.uniform(1, 40)               # dist
            s[base + 1] = self.rng.uniform(0, 3)            # speed
            s[base + 2] = self.rng.uniform(-np.pi, np.pi)   # heading
            s[base + 3:base + 5] = self.rng.uniform(-20, 20, size=2)
        return s

    def _mock_info(self, state, action):
        """Build an info dict with mock VRU + rule signals for the reward."""
        collision = self.rng.random() < 0.02
        info = {
            "action": np.asarray(action, dtype=np.float32),
            "prev_action": self._prev_action.copy(),
            "collision": bool(collision),
            "lane_departure": bool(abs(state[6]) > 1.4),
            "red_light": bool(state[10] == 0 and state[11] < 5 and state[2] > 1),
            "stop_sign_violation": bool(self.rng.random() < 0.01),
            "crosswalk_conflict": float(self.rng.random() < 0.05),
            "general_risk": float(self.rng.uniform(0, 0.3)),
        }
        info["progress"] = float(self.rng.uniform(-0.02, 0.05))
        info["vru_risk"] = compute_vru_risk_target(state, info, self.config)
        return info

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self):
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        if self.mock:
            self._state = self._random_state()
            return self._state.copy()

        # Real mode.  # pragma: no cover - requires CARLA
        self._spawn_actors()
        for _ in range(self.config.fps):
            self.world.tick()
        self._state = self._observe()
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.mock:
            next_state = self._random_state()
            # carry over route progress monotonically for realism
            info = self._mock_info(next_state, action)
            next_state[ROUTE_PROGRESS] = np.clip(
                self._state[ROUTE_PROGRESS] + info["progress"], 0.0, 1.0
            )
            reward = compute_reward(self._state, next_state, info, self.config)
            self._step_count += 1
            done = bool(
                info["collision"]
                or next_state[ROUTE_PROGRESS] >= 1.0
                or self._step_count >= self.config.max_episode_steps
                or self.rng.random() < 0.01
            )
            self._prev_action = action.copy()
            self._state = next_state
            return next_state.copy(), reward, done, info

        # Real mode.  # pragma: no cover - requires CARLA
        self._apply_control(action)
        self.world.tick()
        next_state = self._observe()
        info = self._real_info(next_state, action)
        reward = compute_reward(self._state, next_state, info, self.config)
        self._step_count += 1
        done = bool(
            self._collision_flag
            or next_state[ROUTE_PROGRESS] >= 1.0
            or self._step_count >= self.config.max_episode_steps
        )
        self._prev_action = action.copy()
        self._state = next_state
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
        return s

    def _real_info(self, state, action):  # pragma: no cover - requires CARLA
        info = {
            "action": np.asarray(action, dtype=np.float32),
            "prev_action": self._prev_action.copy(),
            "collision": bool(self._collision_flag),
            "lane_departure": bool(abs(state[6]) > 1.4),
            "red_light": False,
            "stop_sign_violation": False,
            "crosswalk_conflict": 0.0,
            "general_risk": 0.0,
            "progress": float(state[ROUTE_PROGRESS] - self._state[ROUTE_PROGRESS]),
        }
        info["vru_risk"] = compute_vru_risk_target(state, info, self.config)
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
