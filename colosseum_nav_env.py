"""
ColosseumNavEnv — Gymnasium environment wrapping the Colosseum/AirSim
multirotor API for goal-conditioned drone navigation.

Observation space (default):
    Dict with:
        "image"  : uint8 (H, W, 3)  — front RGB camera
        "goal"   : float32 (3,)     — relative target [dx, dy, dz] in metres
        "vel"    : float32 (3,)     — current body velocity [vx, vy, vz]

Action space:
    Box(3,) float32, clipped to [-max_vel, +max_vel] m/s
    Mapped to moveByVelocityAsync calls (NED frame: +X fwd, +Y right, -Z up)

Reward shaping (tunable via constructor kwargs):
    +dist_reward   : progress toward goal (delta distance per step)
    +success_bonus : large reward on arrival within goal_radius
    -collision_pen : large penalty on collision
    -time_penalty  : small negative each step to encourage efficiency
    -smooth_pen    : penalty proportional to raw action change between steps

Action smoothing:
    An EMA filter (action_smooth_alpha) is applied to velocity commands
    before they are sent to the sim, preventing physical jerk regardless
    of policy output.  The reward penalty still tracks raw action deltas
    so the policy learns to output smooth actions as well.

Episode termination:
    - Drone within goal_radius metres of target  → success
    - Collision detected                          → failure
    - Steps exceed max_steps                      → timeout

Stage 3 additions:
    randomize_start : teleport drone to a random XY offset (within
                      start_radius metres) after each takeoff.
    Goal is sampled relative to the actual post-teleport position so
    difficulty stays consistent across all start locations.

Waypoint mode (Stage 3b):
    waypoints : list of (x, y, z) NED positions fixed on the map.
                When provided alongside randomize_goal=True, goals are
                drawn from this list instead of random directions.
                Keeps goals near obstacle height and at known map locations,
                forcing the drone to navigate around structures rather than
                fly in open sky.

Perception capture (Phase 1 — dual-system data collection):
    capture_perception : when True, fetches a depth map (DepthPerspective,
                metres) and a segmentation image (Segmentation, RGB)
                alongside the scene image, and attaches them to the info
                dict under "depth" and "seg".  Used by collect_demos.py to
                build the System 2 (VLM planner) training labels.  Defaults
                to False so PPO training is completely unaffected (no extra
                camera RPCs during oracle/vision/vlm RL).

                All three image types are fetched in a SINGLE simGetImages
                RPC (not three separate calls) and only every
                ``perception_stride`` control ticks. Capturing every step is
                also unnecessary: System 2 only needs a fresh frame at its
                own (slower, ~2Hz) decision rate, while position/velocity
                are still tracked every tick via cheap state queries (no
                image RPC), so future-trajectory and action-chunk labels
                stay fully accurate.

                Command/perception timing: each step() issues the velocity
                command with a generously padded firmware-side `duration`
                (cmd_duration_padding, default 2s) and does NOT block on it
                — instead the loop self-paces to step_duration of real
                elapsed time via time.monotonic()/sleep(), doing the
                (possibly slow) perception RPC inside that window. This
                keeps the commanded velocity continuously active in AirSim
                regardless of how long a perception capture takes, which is
                what prevents the periodic hover/hold "jerk" that occurs if
                the firmware's own duration timer expires before the next
                command is issued. See step()'s inline comments for detail.
"""

from __future__ import annotations

import time
from typing import Any

import airsim
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ── defaults ──────────────────────────────────────────────────────────────────
_DEF_IMG_H = 84
_DEF_IMG_W = 84
_DEF_MAX_VEL = 5.0          # m/s per axis
_DEF_GOAL_RADIUS = 2.0      # metres — "close enough" for success
_DEF_MAX_STEPS = 500
_DEF_STEP_DUR = 0.1         # seconds per action
_DEF_DIST_SCALE = 1.0       # reward scale for distance progress
_DEF_SUCCESS_BONUS = 100.0
_DEF_COLLISION_PEN = -50.0
_DEF_TIME_PEN = -0.05
_DEF_SMOOTH_COEF = 0.1      # penalty weight for action delta between steps
_DEF_ACTION_SMOOTH_ALPHA = 0.6  # EMA weight for outgoing velocity commands (0=frozen, 1=no filter)

# Stuck detection — motor bug-out / physics instability
_DEF_STUCK_CMD_THRESH = 1.0   # min commanded speed (m/s) before we check for stuck
_DEF_STUCK_RATIO = 0.2        # actual / commanded speed below this → stuck vote
_DEF_STUCK_PATIENCE = 10      # consecutive stuck-votes before terminating episode
_DEF_STUCK_PEN = -20.0        # reward penalty for a stuck termination

# Goal randomization defaults (cylindrical bounds relative to start position)
_DEF_GOAL_MIN_DIST = 10.0   # min horizontal distance (m)
_DEF_GOAL_MAX_DIST = 30.0   # max horizontal distance (m)
_DEF_GOAL_ALT_MIN = 5.0     # min altitude above ground (m, NED: stored as negative Z)
_DEF_GOAL_ALT_MAX = 20.0    # max altitude above ground (m)

# Start randomization defaults (Stage 3)
_DEF_START_RADIUS = 0.0     # 0 = no randomization; set >0 to enable


class ColosseumNavEnv(gym.Env):
    """Goal-conditioned 3-D navigation in Colosseum/AirSim."""

    metadata = {"render_modes": ["rgb_array"]}

    # ── initialise ────────────────────────────────────────────────────────────
    def __init__(
        self,
        goal: tuple[float, float, float] = (20.0, 0.0, -10.0),
        img_h: int = _DEF_IMG_H,
        img_w: int = _DEF_IMG_W,
        max_vel: float = _DEF_MAX_VEL,
        goal_radius: float = _DEF_GOAL_RADIUS,
        max_steps: int = _DEF_MAX_STEPS,
        step_duration: float = _DEF_STEP_DUR,
        dist_scale: float = _DEF_DIST_SCALE,
        success_bonus: float = _DEF_SUCCESS_BONUS,
        collision_pen: float = _DEF_COLLISION_PEN,
        time_pen: float = _DEF_TIME_PEN,
        smooth_coef: float = _DEF_SMOOTH_COEF,
        action_smooth_alpha: float = _DEF_ACTION_SMOOTH_ALPHA,
        randomize_goal: bool = False,
        goal_min_dist: float = _DEF_GOAL_MIN_DIST,
        goal_max_dist: float = _DEF_GOAL_MAX_DIST,
        goal_alt_min: float = _DEF_GOAL_ALT_MIN,
        goal_alt_max: float = _DEF_GOAL_ALT_MAX,
        waypoints: list[tuple[float, float, float]] | None = None,
        randomize_start: bool = False,
        start_radius: float = _DEF_START_RADIUS,
        stuck_cmd_thresh: float = _DEF_STUCK_CMD_THRESH,
        stuck_ratio: float = _DEF_STUCK_RATIO,
        stuck_patience: int = _DEF_STUCK_PATIENCE,
        stuck_pen: float = _DEF_STUCK_PEN,
        camera_name: str = "0",
        vehicle_name: str = "",
        render_mode: str | None = None,
        include_image: bool = True,
        capture_perception: bool = False,
        perception_stride: int = 5,
        cmd_duration_padding: float = 2.0,
        depth_clip: float = 100.0,
        collision_grace_steps: int = 3,
    ) -> None:
        super().__init__()

        self._fixed_goal = np.array(goal, dtype=np.float32)  # used when not randomizing
        self.goal_world = self._fixed_goal.copy()            # NED metres; updated each reset if randomizing
        self.img_h = img_h
        self.img_w = img_w
        self.max_vel = max_vel
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.step_duration = step_duration
        self.dist_scale = dist_scale
        self.success_bonus = success_bonus
        self.collision_pen = collision_pen
        self.time_pen = time_pen
        self.smooth_coef = smooth_coef
        self.action_smooth_alpha = action_smooth_alpha
        self.randomize_goal = randomize_goal
        self.goal_min_dist = goal_min_dist
        self.goal_max_dist = goal_max_dist
        self.goal_alt_min = goal_alt_min
        self.goal_alt_max = goal_alt_max
        self.waypoints: list[np.ndarray] = (
            [np.array(wp, dtype=np.float32) for wp in waypoints]
            if waypoints else []
        )
        self.randomize_start = randomize_start
        self.start_radius = start_radius
        self.stuck_cmd_thresh = stuck_cmd_thresh
        self.stuck_ratio = stuck_ratio
        self.stuck_patience = stuck_patience
        self.stuck_pen = stuck_pen
        self.camera_name = camera_name
        self.vehicle_name = vehicle_name
        self.render_mode = render_mode
        self.include_image = include_image
        self.capture_perception = capture_perception
        self.perception_stride = max(1, perception_stride)
        self.cmd_duration_padding = cmd_duration_padding
        self.depth_clip = depth_clip
        self.collision_grace_steps = collision_grace_steps
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._smoothed_cmd = np.zeros(3, dtype=np.float32)
        self._stuck_steps: int = 0
        self._last_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        self._last_depth: np.ndarray | None = None
        self._last_seg: np.ndarray | None = None

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(img_h, img_w, 3),
                dtype=np.uint8,
            ),
            "goal": spaces.Box(
                low=-500.0, high=500.0,
                shape=(3,),
                dtype=np.float32,
            ),
            "vel": spaces.Box(
                low=-50.0, high=50.0,
                shape=(3,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-max_vel, high=max_vel,
            shape=(3,),
            dtype=np.float32,
        )

        # AirSim client — connected lazily on first reset
        self._client: airsim.MultirotorClient | None = None
        self._step_count: int = 0
        self._prev_dist: float = 0.0
        self._collision_ts_at_reset: float = 0.0

    # ── internal helpers ──────────────────────────────────────────────────────
    def _connect(self) -> airsim.MultirotorClient:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True, vehicle_name=self.vehicle_name)
        client.armDisarm(True, vehicle_name=self.vehicle_name)
        return client

    def _get_position(self) -> np.ndarray:
        state = self._client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _get_velocity(self) -> np.ndarray:
        state = self._client.getMultirotorState(vehicle_name=self.vehicle_name)
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)

    # ── image decode helpers (shared by single-type and combined RPCs) ────────
    def _decode_scene(self, resp) -> np.ndarray:
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img.size == 0 or resp.height == 0 or resp.width == 0:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        img = img.reshape(resp.height, resp.width, 3)
        if img.shape[:2] != (self.img_h, self.img_w):
            img = cv2.resize(img, (self.img_w, self.img_h))
        return img

    def _decode_depth(self, resp) -> np.ndarray:
        """Depth in metres. Sky/far pixels clipped, inf/nan sanitised."""
        if resp.height == 0 or resp.width == 0 or len(resp.image_data_float) == 0:
            return np.zeros((self.img_h, self.img_w), dtype=np.float32)
        depth = np.array(resp.image_data_float, dtype=np.float32)
        depth = depth.reshape(resp.height, resp.width)
        depth = np.nan_to_num(depth, nan=self.depth_clip,
                              posinf=self.depth_clip, neginf=0.0)
        np.clip(depth, 0.0, self.depth_clip, out=depth)
        return depth

    def _decode_seg(self, resp) -> np.ndarray:
        """Segmentation RGB (per-object colours), native render resolution."""
        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img.size == 0 or resp.height == 0 or resp.width == 0:
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        return img.reshape(resp.height, resp.width, 3)

    def _get_image(self) -> np.ndarray:
        responses = self._client.simGetImages([
            airsim.ImageRequest(
                self.camera_name, airsim.ImageType.Scene,
                pixels_as_float=False, compress=False,
            )
        ], vehicle_name=self.vehicle_name)
        return self._decode_scene(responses[0])

    def _get_depth(self) -> np.ndarray:
        responses = self._client.simGetImages([
            airsim.ImageRequest(
                self.camera_name, airsim.ImageType.DepthPerspective,
                pixels_as_float=True, compress=False,
            )
        ], vehicle_name=self.vehicle_name)
        return self._decode_depth(responses[0])

    def _get_segmentation(self) -> np.ndarray:
        responses = self._client.simGetImages([
            airsim.ImageRequest(
                self.camera_name, airsim.ImageType.Segmentation,
                pixels_as_float=False, compress=False,
            )
        ], vehicle_name=self.vehicle_name)
        return self._decode_seg(responses[0])

    def _refresh_perception_cache(self) -> None:
        """Fetch the scene image, and on capture ticks also depth + segmentation,
        updating self._last_image / _last_depth / _last_seg in place.

        Behaviour by mode:
          - capture_perception=False (oracle/vision/vlm RL): unchanged from
            before — fetch the scene image every step via a single RPC,
            never touch depth/seg. No regression for existing PPO training.
          - capture_perception=True (collect_demos.py): all three image
            types are fetched together in ONE simGetImages RPC, and only
            every ``perception_stride`` steps. Off-stride steps reuse the
            cached last frame with no RPC at all.

        IMPORTANT: callers (step()) must invoke this WHILE the previous
        moveByVelocityAsync future is still in flight, i.e. *before*
        `.join()`-ing it, not after. AirSim's velocity commands only hold
        for their commanded `duration`; if the (slow, ~150-500ms) combined
        image RPC runs strictly *after* the move already finished and was
        joined, the drone sits with no active command for that entire
        window and the flight controller's hover/hold reasserts itself —
        producing a visible position correction "jerk" once the next
        velocity command resumes. Issuing the image RPC *before* joining
        the move future overlaps the two so the maneuver is still being
        executed by the sim backend throughout the capture, eliminating
        that dead-time gap entirely.
        """
        if not self.include_image and not self.capture_perception:
            return

        if not self.capture_perception:
            self._last_image = self._get_image()
            return

        self._last_depth = None
        self._last_seg = None
        if (self._step_count % self.perception_stride) != 0:
            return

        responses = self._client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene,
                                 pixels_as_float=False, compress=False),
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective,
                                 pixels_as_float=True, compress=False),
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Segmentation,
                                 pixels_as_float=False, compress=False),
        ], vehicle_name=self.vehicle_name)
        self._last_image = self._decode_scene(responses[0])
        self._last_depth = self._decode_depth(responses[1])
        self._last_seg = self._decode_seg(responses[2])

    def _perception_info(self) -> dict[str, np.ndarray] | None:
        """Depth + segmentation for the info dict; None on non-capture ticks."""
        if self._last_depth is None or self._last_seg is None:
            return None
        return {"depth": self._last_depth, "seg": self._last_seg}

    def _snapshot_collision_baseline(self) -> None:
        """Record collision timestamp after reset/takeoff.

        AirSim keeps ``has_collided=True`` until the flag is read-and-reset
        server-side (not exposed in Python). Takeoff/landing often set the
        flag once; we only treat *new* collisions (later timestamp) as real.
        """
        info = self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        self._collision_ts_at_reset = info.time_stamp

    def _check_collision(self) -> bool:
        if self._step_count <= self.collision_grace_steps:
            return False
        info = self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        return info.has_collided and info.time_stamp > self._collision_ts_at_reset

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build the obs dict from current state + the perception cache.

        Does NOT trigger any image RPC itself — callers must call
        _refresh_perception_cache() at the appropriate point beforehand
        (see step()'s overlap-with-move-future comment).
        """
        pos = self._get_position()
        rel_goal = (self.goal_world - pos).astype(np.float32)
        vel = self._get_velocity()
        return {"image": self._last_image, "goal": rel_goal, "vel": vel}

    def _dist_to_goal(self, pos: np.ndarray | None = None) -> float:
        if pos is None:
            pos = self._get_position()
        return float(np.linalg.norm(self.goal_world - pos))

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)

        # Connect / reconnect
        if self._client is None:
            self._client = self._connect()

        # Reset sim to known state
        self._client.reset()
        time.sleep(0.3)                # let physics settle
        self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self._client.armDisarm(True, vehicle_name=self.vehicle_name)
        self._client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.5)                # let takeoff settle before first action
        self._snapshot_collision_baseline()

        self._step_count = 0
        self._stuck_steps = 0
        self._smoothed_cmd = np.zeros(3, dtype=np.float32)
        # Seed prev_action from actual velocity so step 1 doesn't penalize
        # the first movement as if the drone snapped from standstill.
        self._prev_action = self._get_velocity()

        # ── Stage 3: random start position ────────────────────────────────────
        if self.randomize_start and self.start_radius > 0:
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            r = self.np_random.uniform(0.0, self.start_radius)
            # Mutate current pose in-place (preserves orientation from takeoff)
            cur_pose = self._client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            cur_pose.position.x_val += r * np.cos(angle)
            cur_pose.position.y_val += r * np.sin(angle)
            self._client.simSetVehiclePose(
                cur_pose, ignore_collision=True, vehicle_name=self.vehicle_name
            )
            time.sleep(0.3)
            self._snapshot_collision_baseline()

        # ── Goal sampling ──────────────────────────────────────────────────────
        if self.randomize_goal:
            if self.waypoints:
                # Waypoint mode: pick a random fixed map location.
                # Exclude any waypoint that is too close to current start.
                start_pos = self._get_position()
                candidates = [
                    wp for wp in self.waypoints
                    if np.linalg.norm(wp[:2] - start_pos[:2]) >= self.goal_min_dist
                ]
                if not candidates:
                    candidates = self.waypoints  # fallback: all waypoints
                idx = int(self.np_random.integers(0, len(candidates)))
                self.goal_world = candidates[idx].copy()
            else:
                # Random direction mode: sample relative to actual start position.
                start_pos = self._get_position()
                angle = self.np_random.uniform(0.0, 2.0 * np.pi)
                h_dist = self.np_random.uniform(self.goal_min_dist, self.goal_max_dist)
                alt = self.np_random.uniform(self.goal_alt_min, self.goal_alt_max)
                self.goal_world = np.array([
                    start_pos[0] + h_dist * np.cos(angle),
                    start_pos[1] + h_dist * np.sin(angle),
                    -alt,               # NED: negative Z = up (absolute altitude)
                ], dtype=np.float32)

        self._prev_dist = self._dist_to_goal()

        self._refresh_perception_cache()
        obs = self._build_obs()
        info: dict = {"dist_to_goal": self._prev_dist, "goal": self.goal_world.tolist()}
        perception = self._perception_info()
        if perception is not None:
            info.update(perception)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        assert self._client is not None, "Call reset() before step()."

        # Clip raw policy action
        raw_action = np.clip(action, -self.max_vel, self.max_vel).astype(np.float32)

        # EMA filter: smooth the command sent to the sim to prevent physical jerk.
        # The reward penalty still tracks raw_action so the policy learns to be
        # smooth in its own outputs too.
        self._smoothed_cmd = (
            self.action_smooth_alpha * raw_action
            + (1.0 - self.action_smooth_alpha) * self._smoothed_cmd
        )
        vx, vy, vz = self._smoothed_cmd.tolist()
        # NOTE on timing: `duration` here is a generous SAFETY CEILING
        # (cmd_duration_padding, default 2s) for AirSim's firmware-side
        # velocity command — it is deliberately decoupled from the env's
        # logical control-rate clock, which remains step_duration via the
        # explicit self-paced sleep below.
        #
        # Why: AirSim auto-reverts to hover/hold once `duration` elapses,
        # regardless of what Python is doing. If we left `duration` at
        # step_duration (0.1s) the firmware would revert mid-capture
        # whenever perception capture takes longer than 0.1s (likely, for a
        # combined scene+depth+seg RPC) — causing a brief "catch up" jerk on
        # the next command even with the overlap below.
        #
        # We deliberately do NOT call `.join()` on this future: joining
        # blocks until the maneuver's full `duration` elapses, so if we
        # padded duration to 2s *and* joined, every step would take ~2s.
        # Instead we self-pace with time.monotonic()/sleep() to
        # step_duration of real elapsed time below, while the next step()
        # call's fresh command naturally overrides this one (well within its
        # padded safety window) to keep flight continuous.
        t_cmd_issued = time.monotonic()
        self._client.moveByVelocityAsync(
            vx, vy, vz,
            duration=max(self.step_duration, self.cmd_duration_padding),
            vehicle_name=self.vehicle_name,
        )

        self._step_count += 1

        # Fetch perception WHILE the move above is active in the sim backend
        # (overlap) — see _refresh_perception_cache docstring. This is what
        # prevents the periodic directional jerk: the commanded velocity
        # stays active throughout the capture instead of expiring first.
        self._refresh_perception_cache()

        # Self-pace to step_duration of real elapsed time so the physics
        # integrates the expected amount of motion before we read state.
        # On capture ticks, perception alone often consumes the whole
        # budget (remaining <= 0) and we proceed immediately; on cheap
        # cached ticks we top up the difference with a short sleep.
        remaining = self.step_duration - (time.monotonic() - t_cmd_issued)
        if remaining > 0:
            time.sleep(remaining)

        # Gather state (after ~step_duration of real elapsed time, same
        # reward/observation timing semantics as before this change)
        pos = self._get_position()
        cur_dist = self._dist_to_goal(pos)
        vel = self._get_velocity()
        collision = self._check_collision()

        # ── Stuck detection (motor bug-out / physics instability) ─────────────
        # Only active outside the takeoff grace window. Votes "stuck" when we
        # are commanding meaningful speed but the drone barely moves.
        stuck = False
        if self._step_count > self.collision_grace_steps:
            cmd_speed = float(np.linalg.norm(self._smoothed_cmd))
            actual_speed = float(np.linalg.norm(vel))
            if (cmd_speed >= self.stuck_cmd_thresh and
                    actual_speed < self.stuck_ratio * cmd_speed):
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
            stuck = self._stuck_steps >= self.stuck_patience

        # Reward (tracks raw action delta, not smoothed, to teach the policy itself)
        dist_delta = self._prev_dist - cur_dist          # positive = closer
        action_delta = float(np.linalg.norm(raw_action - self._prev_action))
        reward = (
            self.dist_scale * dist_delta
            + self.time_pen
            - self.smooth_coef * action_delta
        )
        self._prev_action = raw_action.copy()

        terminated = False
        truncated = False

        if cur_dist < self.goal_radius:
            reward += self.success_bonus
            terminated = True
        elif stuck:
            reward += self.stuck_pen
            terminated = True
        elif collision:
            reward += self.collision_pen
            terminated = True
        elif self._step_count >= self.max_steps:
            truncated = True

        self._prev_dist = cur_dist

        obs = self._build_obs()
        info = {
            "dist_to_goal": cur_dist,
            "collision": collision,
            "stuck": stuck,
            "step": self._step_count,
            "goal": self.goal_world.tolist(),
        }
        perception = self._perception_info()
        if perception is not None:
            info.update(perception)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._get_image()
        return None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.landAsync(vehicle_name=self.vehicle_name).join()
                self._client.armDisarm(False, vehicle_name=self.vehicle_name)
                self._client.enableApiControl(
                    False, vehicle_name=self.vehicle_name
                )
            except Exception:
                pass
            self._client = None
