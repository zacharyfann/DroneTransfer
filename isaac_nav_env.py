"""
isaac_nav_env.py

Gymnasium bridge that adapts Isaac Sim/Isaac Lab tasks to the same
goal-conditioned interface used by ColosseumNavEnv:
    obs = {"image": HxWx3 uint8, "goal": (3,), "vel": (3,)}
    action = (vx, vy, vz) clipped to [-max_vel, max_vel]

This wrapper is intentionally defensive because Isaac tasks vary in their
observation dictionaries. If explicit goal/vel/image keys are missing, it
falls back to a policy/state vector and uses the first 6 values as
[goal(3), vel(3)].
"""

from __future__ import annotations

import argparse
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None

_SIM_APP = None


def _ensure_isaac_lab(*, headless: bool = True, livestream: int = -1, enable_cameras: bool = False) -> None:
    """Launch Isaac Sim once and register Isaac Lab gym tasks.

    In Docker there is no display — use ``livestream=2`` (not headless=False)
    when you want WebRTC viewing via ``--isaac-vis``.
    """
    global _SIM_APP
    if _SIM_APP is not None:
        return

    import os

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    known_args = {a.dest for a in parser._actions if a.dest != "help"}

    # Corp/air-gapped hosts cannot reach NVIDIA extension registries; use bundled exts only.
    registry_off = "--/app/extensions/registryEnabled=false"

    if livestream > 0:
        os.environ["ENABLE_CAMERAS"] = "1"
        # Isaac Sim WebRTC flags differ by generation:
        # - legacy (5.1): --/app/livestream/publicEndpointAddress + --/app/livestream/port
        # - modern (6.x): --/exts/omni.kit.livestream.app/primaryStream/*
        # Select via ISAAC_WEBRTC_STYLE={legacy,modern}. Default is modern.
        stream_ip = os.environ.get("PUBLIC_IP", "127.0.0.1")
        os.environ.setdefault("PUBLIC_IP", stream_ip)
        webrtc_style = os.environ.get("ISAAC_WEBRTC_STYLE", "modern").lower()
        if webrtc_style == "legacy":
            webrtc_args = (
                f"--/app/livestream/publicEndpointAddress={stream_ip} "
                "--/app/livestream/port=49100"
            )
        else:
            webrtc_args = (
                f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={stream_ip} "
                "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 "
                "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998"
            )
        defaults = {
            "headless": True,
            "livestream": livestream,
            "enable_cameras": enable_cameras,
            "kit_args": f"{webrtc_args} {registry_off}",
            "visualizer": ["kit"],  # Isaac Lab 3.x / develop only
        }
    else:
        defaults = {
            "headless": headless,
            "livestream": -1,
            "kit_args": registry_off,
            "visualizer": ["none"],  # Isaac Lab 3.x / develop only
        }
    parser.set_defaults(**{k: v for k, v in defaults.items() if k in known_args})
    args_cli, _ = parser.parse_known_args([])
    app_launcher = AppLauncher(args_cli)
    _SIM_APP = app_launcher.app

    import isaaclab_tasks  # noqa: F401 — registers gym env ids


def _to_numpy(x: Any) -> Any:
    """Convert torch tensors (possibly on GPU) to numpy recursively."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    return x


def _flatten_vec(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def _resolve_task_ids(task_id: str) -> list[str]:
    """Return task id candidates for Isaac Lab develop vs main naming."""
    if task_id.endswith("-v0"):
        return [task_id[:-3], task_id]
    # develop branch: unversioned ids (e.g. Isaac-Cartpole-Direct)
    return [task_id, f"{task_id}-v0"]


def _make_isaac_task(
    task_id: str,
    num_envs: int = 1,
    device: str = "cuda:0",
    *,
    visual_friendly: bool = False,
) -> gym.Env:
    """Create an Isaac Lab gym env with the required cfg object."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    last_err: Exception | None = None
    for tid in _resolve_task_ids(task_id):
        try:
            env_cfg = parse_env_cfg(tid, device=device, num_envs=num_envs)
            if visual_friendly and hasattr(env_cfg, "scene"):
                # Fabric clones often don't show in the Kit/WebRTC viewport.
                env_cfg.scene.clone_in_fabric = False
                env_cfg.scene.replicate_physics = False
                if hasattr(env_cfg, "debug_vis"):
                    env_cfg.debug_vis = False  # goal markers use Fabric instancers → black WebRTC
                print(
                    "[isaac_nav_env] visual mode: clone_in_fabric=False, "
                    "replicate_physics=False, debug_vis=False"
                )
            return gym.make(tid, cfg=env_cfg)
        except gym.error.DeprecatedEnv:
            if tid.endswith("-v0"):
                tid = tid[:-3]
                try:
                    env_cfg = parse_env_cfg(tid, device=device, num_envs=num_envs)
                    if visual_friendly and hasattr(env_cfg, "scene"):
                        env_cfg.scene.clone_in_fabric = False
                        env_cfg.scene.replicate_physics = False
                        if hasattr(env_cfg, "debug_vis"):
                            env_cfg.debug_vis = False
                        print(
                            "[isaac_nav_env] visual mode: clone_in_fabric=False, "
                            "replicate_physics=False, debug_vis=False"
                        )
                    return gym.make(tid, cfg=env_cfg)
                except Exception as err:  # noqa: BLE001
                    last_err = err
        except Exception as err:  # noqa: BLE001
            last_err = err

    if last_err is not None:
        raise last_err
    raise gym.error.NameNotFound(f"No registered env matching {task_id!r}")


class IsaacNavEnv(gym.Env):
    """Adapter from an Isaac gym task to DroneTransfer's nav interface."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: str,
        img_h: int = 84,
        img_w: int = 84,
        max_vel: float = 5.0,
        goal_radius: float = 2.0,
        max_steps: int = 500,
        headless: bool = True,
        render_mode: str | None = None,
        include_image: bool = True,
        num_envs: int = 1,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.img_h = img_h
        self.img_w = img_w
        self.max_vel = max_vel
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.headless = headless
        self.render_mode = render_mode
        self.include_image = include_image

        self._step_count = 0
        self._last_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        _ensure_isaac_lab(
            headless=headless,
            livestream=2 if not headless else -1,
        )

        self._base = _make_isaac_task(task_id, num_envs=num_envs, device=device)
        self.task_id = getattr(self._base.spec, "id", task_id) if self._base.spec else task_id

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(img_h, img_w, 3),
                    dtype=np.uint8,
                ),
                "goal": spaces.Box(
                    low=-500.0,
                    high=500.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "vel": spaces.Box(
                    low=-100.0,
                    high=100.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-max_vel, high=max_vel, shape=(3,), dtype=np.float32
        )

    def _extract_goal_vel_image(self, raw_obs: Any) -> dict[str, np.ndarray]:
        obs_np = _to_numpy(raw_obs)

        goal = np.zeros(3, dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        image = self._last_image

        if isinstance(obs_np, dict):
            # Common task-specific key names first.
            if "goal" in obs_np:
                goal = _flatten_vec(obs_np["goal"])[:3]
            if "vel" in obs_np:
                vel = _flatten_vec(obs_np["vel"])[:3]

            # Fallback: parse from policy/state vector.
            if np.allclose(goal, 0.0) or np.allclose(vel, 0.0):
                if "policy" in obs_np:
                    vec = _flatten_vec(obs_np["policy"])
                elif "state" in obs_np:
                    vec = _flatten_vec(obs_np["state"])
                elif "observation" in obs_np:
                    vec = _flatten_vec(obs_np["observation"])
                else:
                    vec = None
                if vec is not None and vec.size >= 6:
                    goal = vec[:3].astype(np.float32)
                    vel = vec[3:6].astype(np.float32)

            # Optional camera keys.
            if self.include_image:
                for key in ("image", "rgb", "camera", "rgb_image"):
                    if key in obs_np:
                        img = np.asarray(obs_np[key])
                        if img.ndim == 4:  # [N,H,W,C], use first env
                            img = img[0]
                        if img.ndim == 3 and img.shape[-1] in (3, 4):
                            if img.shape[-1] == 4:
                                img = img[..., :3]
                            if img.dtype != np.uint8:
                                # Map [0,1] floats or generic numeric to uint8.
                                if np.issubdtype(img.dtype, np.floating):
                                    img = np.clip(img, 0.0, 1.0) * 255.0
                                img = img.astype(np.uint8)
                            if img.shape[:2] != (self.img_h, self.img_w):
                                img = cv2.resize(img, (self.img_w, self.img_h))
                            image = img
                            break
        else:
            vec = _flatten_vec(obs_np)
            if vec.size >= 6:
                goal = vec[:3].astype(np.float32)
                vel = vec[3:6].astype(np.float32)

        self._last_image = image
        return {"image": image, "goal": goal.astype(np.float32), "vel": vel.astype(np.float32)}

    def _map_action(self, action: np.ndarray) -> Any:
        cmd = np.clip(np.asarray(action, dtype=np.float32), -self.max_vel, self.max_vel)
        base_space = self._base.action_space

        if isinstance(base_space, spaces.Box):
            mapped = np.zeros(base_space.shape, dtype=np.float32)
            flat = mapped.reshape(-1)
            flat[: min(3, flat.size)] = cmd[: min(3, flat.size)]
            mapped = mapped.reshape(base_space.shape)
            low = np.asarray(base_space.low, dtype=np.float32)
            high = np.asarray(base_space.high, dtype=np.float32)
            return np.clip(mapped, low, high)

        if isinstance(base_space, spaces.Discrete):
            # If task is discrete, map velocity intent to one of three bins.
            axis = int(np.argmax(np.abs(cmd)))
            if cmd[axis] > 0:
                return min(base_space.n - 1, 1)
            if cmd[axis] < 0:
                return min(base_space.n - 1, 2)
            return 0

        return cmd

    def _isaac_device(self) -> str:
        env = self._base.unwrapped if hasattr(self._base, "unwrapped") else self._base
        return str(getattr(env, "device", "cuda:0"))

    def _to_isaac_action(self, action: Any) -> Any:
        """Isaac Lab 8.x DirectRLEnv expects torch actions on the sim device."""
        if isinstance(action, (int, np.integer)):
            return int(action)
        if torch is None:
            return action

        arr = np.asarray(action, dtype=np.float32)
        env = self._base.unwrapped if hasattr(self._base, "unwrapped") else self._base
        num_envs = int(getattr(env, "num_envs", 1))

        if arr.ndim == 0:
            arr = arr.reshape(num_envs, 1)
        elif arr.ndim == 1:
            if arr.shape[0] == num_envs:
                arr = arr.reshape(num_envs, -1)
            else:
                arr = np.broadcast_to(arr, (num_envs, arr.shape[0])).copy()

        return torch.as_tensor(arr, device=self._isaac_device(), dtype=torch.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        self._step_count = 0
        out = self._base.reset(seed=seed, options=options)
        if isinstance(out, tuple) and len(out) == 2:
            raw_obs, info = out
        else:
            raw_obs, info = out, {}
        obs = self._extract_goal_vel_image(raw_obs)
        dist = float(np.linalg.norm(obs["goal"]))
        info = dict(info)
        info["dist_to_goal"] = dist
        info["collision"] = False
        info["stuck"] = False
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self._step_count += 1
        mapped_action = self._to_isaac_action(self._map_action(action))
        raw_obs, reward, terminated, truncated, info = self._base.step(mapped_action)
        obs = self._extract_goal_vel_image(raw_obs)
        dist = float(np.linalg.norm(obs["goal"]))

        # Keep same success semantics as Colosseum env.
        if dist < self.goal_radius:
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        if isinstance(reward, (np.ndarray, list, tuple)):
            reward = float(np.asarray(reward).reshape(-1)[0])
        else:
            reward = float(reward)

        if isinstance(terminated, (np.ndarray, list, tuple)):
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
        else:
            terminated = bool(terminated)

        if isinstance(truncated, (np.ndarray, list, tuple)):
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
        else:
            truncated = bool(truncated)

        info = dict(info) if isinstance(info, dict) else {}
        info["dist_to_goal"] = dist
        info.setdefault("collision", False)
        info.setdefault("stuck", False)
        info["step"] = self._step_count
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._last_image
        return None

    def close(self) -> None:
        self._base.close()
