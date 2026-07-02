"""
collect_demos.py — Phase 1 of the dual-system (VLM + diffusion) pipeline.

Runs the trained *oracle* PPO policy in ColosseumNavEnv with the camera and
perception capture turned ON, and records a densely-labelled demonstration
dataset for training System 2 (the VLM planner) and System 1 (the action
expert).

The oracle only *acts* on [goal, vel] (it is blind), but we record the full
sensory context at every 10 Hz control tick, plus two hindsight labels:

  • future_offsets — where the drone ACTUALLY went over the next ~3 s,
                     relative to its current position (System 2 trajectory
                     target).
  • action_chunk   — the oracle's next H velocity commands (System 1 target).

Each saved sample:
    image          uint8  (img_size, img_size, 3)   front RGB  (System 2 input)
    goal           f32    (3,)                       relative goal [dx,dy,dz]
    vel            f32    (3,)                       body velocity
    depth          f32    (P, P)                     metric depth (System 2 label)
    seg            uint8  (P, P, 3)                  segmentation RGB (System 2 label)
    future_offsets f32    (K, 3)                     +0.5 .. +3.0 s positions − now
    future_mask    u8     (K,)                        1 = valid, 0 = past episode end
    action_chunk   f32    (H, 3)                     next H oracle actions
    action_mask    u8     (H,)                        1 = valid, 0 = padded

Output: sharded, compressed .npz files in --out, plus a meta.json describing
the collection config.

Usage:
  python collect_demos.py --checkpoint checkpoints/oracle/ppo_oracle_414240_steps `
      --steps 20000 --out data/demos --waypoints-file waypoints.json --device cuda

Image/depth/segmentation are fetched in a single combined RPC and only every
``perception_stride`` control ticks (default 5, ~2Hz) — NOT every 10Hz tick.
Position/velocity are still tracked every tick via cheap state queries, so
future-trajectory and action-chunk labels remain fully accurate. This design
exists because issuing 3 separate blocking image RPCs every single 0.1s
control step stalls the velocity command loop well past its duration —
the drone visibly hovers/jerks between commands — and corrupts the very
labels we're trying to record. Capturing less often (matching System 2's
real decision rate) also avoids the AirSim/UE memory growth seen from
sustained high-frequency multi-image-type capture over long collection runs.

NOTE: this is pipeline-validation data from the BLIND oracle. It teaches smooth
goal-seeking + scene understanding, NOT true obstacle avoidance — that requires
the privileged-planner teacher in a later phase.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import cv2
import numpy as np
from stable_baselines3 import PPO

from colosseum_nav_env import ColosseumNavEnv


# ─────────────────────────────────────────────────────────────────────────────
# Dataset shard writer
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Accumulates samples in memory and flushes compressed .npz shards."""

    _KEYS = (
        "image", "goal", "vel", "depth", "seg",
        "future_offsets", "future_mask", "action_chunk", "action_mask",
    )

    def __init__(self, out_dir: str, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self._buf: dict[str, list] = {k: [] for k in self._KEYS}
        self._shard_idx = 0
        self._total = 0

    def add(self, sample: dict[str, np.ndarray]) -> None:
        for k in self._KEYS:
            self._buf[k].append(sample[k])
        self._total += 1
        if len(self._buf["image"]) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        n = len(self._buf["image"])
        if n == 0:
            return
        path = os.path.join(self.out_dir, f"shard_{self._shard_idx:05d}.npz")
        arrays = {k: np.stack(self._buf[k], axis=0) for k in self._KEYS}
        np.savez_compressed(path, **arrays)
        print(f"  wrote {path}  ({n} samples)")
        self._shard_idx += 1
        self._buf = {k: [] for k in self._KEYS}

    @property
    def total(self) -> int:
        return self._total


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pos_from(info_goal: list, rel_goal: np.ndarray) -> np.ndarray:
    """Absolute NED position = goal_world − relative_goal."""
    return np.asarray(info_goal, dtype=np.float32) - rel_goal.astype(np.float32)


def _resize_depth(depth: np.ndarray, size: int) -> np.ndarray:
    if depth.shape[:2] == (size, size):
        return depth.astype(np.float32)
    return cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _resize_seg(seg: np.ndarray, size: int) -> np.ndarray:
    if seg.shape[:2] == (size, size):
        return seg.astype(np.uint8)
    return cv2.resize(seg, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def process_episode(
    decisions: list[dict[str, Any]],
    positions: np.ndarray,
    actions: np.ndarray,
    writer: ShardWriter,
    *,
    horizon_k: int,
    stride: int,
    chunk_h: int,
    perception_size: int,
) -> None:
    """Apply hindsight relabelling to one episode and push samples to writer.

    ``positions``/``actions`` cover EVERY control tick of the episode (cheap
    state queries, no image RPCs) so trajectory/action-chunk labels stay
    accurate regardless of how sparsely images were captured.

    ``decisions`` holds only the ticks where a scene image (+ depth/seg) was
    actually fetched — each entry has keys: image, goal, vel, depth, seg, and
    "t" (its index into positions/actions).
    """
    n = positions.shape[0]

    for d in decisions:
        t = d["t"]

        # Future trajectory targets (relative offsets) at +stride, +2*stride, ...
        future_offsets = np.zeros((horizon_k, 3), dtype=np.float32)
        future_mask = np.zeros((horizon_k,), dtype=np.uint8)
        for k in range(1, horizon_k + 1):
            idx = t + k * stride
            if idx < n:
                future_offsets[k - 1] = positions[idx] - positions[t]
                future_mask[k - 1] = 1

        # Action chunk targets (next H oracle commands)
        action_chunk = np.zeros((chunk_h, 3), dtype=np.float32)
        action_mask = np.zeros((chunk_h,), dtype=np.uint8)
        for h in range(chunk_h):
            idx = t + h
            if idx < n:
                action_chunk[h] = actions[idx]
                action_mask[h] = 1

        writer.add({
            "image":          d["image"],
            "goal":           d["goal"].astype(np.float32),
            "vel":            d["vel"].astype(np.float32),
            "depth":          _resize_depth(d["depth"], perception_size),
            "seg":            _resize_seg(d["seg"], perception_size),
            "future_offsets": future_offsets,
            "future_mask":    future_mask,
            "action_chunk":   action_chunk,
            "action_mask":    action_mask,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect(args: argparse.Namespace) -> None:
    waypoints = None
    if args.waypoints_file:
        with open(args.waypoints_file) as f:
            waypoints = json.load(f)
        print(f"Waypoints     : {len(waypoints)} from {args.waypoints_file}")

    print(f"Oracle ckpt   : {args.checkpoint}")
    print(f"Target samples: {args.steps:,}  (decision-point samples, not control ticks)")
    print(f"Scene image   : {args.img_size}x{args.img_size}")
    print(f"Perception    : {args.perception_size}x{args.perception_size}  "
          f"every {args.perception_stride} control ticks (~{10.0/args.perception_stride:.1f}Hz)")
    print(f"Traj horizon  : K={args.horizon_k} @ stride={args.stride} "
          f"({args.horizon_k * args.stride * 0.1:.1f}s)")
    print(f"Action chunk  : H={args.chunk_h}")
    print(f"Out dir       : {args.out}\n")

    model = PPO.load(args.checkpoint.removesuffix(".zip"), device=args.device)

    env = ColosseumNavEnv(
        img_h=args.img_size,
        img_w=args.img_size,
        max_vel=5.0,
        goal_radius=2.0,
        max_steps=args.max_ep_steps,
        step_duration=0.1,
        smooth_coef=0.05,
        action_smooth_alpha=0.6,
        randomize_goal=True,
        goal_min_dist=args.goal_min_dist,
        goal_max_dist=args.goal_max_dist,
        randomize_start=True,
        start_radius=args.start_radius,
        stuck_patience=7,
        waypoints=waypoints,
        include_image=True,                      # need the camera now
        capture_perception=True,                 # depth + seg into info
        perception_stride=args.perception_stride,  # only fetch images every N ticks
    )

    writer = ShardWriter(args.out, args.shard_size)
    rng_seed = args.seed
    episodes = 0

    while writer.total < args.steps:
        obs, info = env.reset(seed=rng_seed)
        rng_seed += 1
        positions: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        decisions: list[dict[str, Any]] = []
        done = False

        while not done:
            rel_goal = obs["goal"].astype(np.float32)
            vel = obs["vel"].astype(np.float32)
            oracle_obs = np.concatenate([rel_goal, vel], axis=0)
            action, _ = model.predict(oracle_obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            t = len(positions)
            positions.append(_pos_from(info["goal"], rel_goal))
            actions.append(action)

            # "depth" is only present in info on a perception-capture tick
            # (see ColosseumNavEnv.perception_stride) — cheap on every other tick.
            if "depth" in info:
                decisions.append({
                    "t":     t,
                    "image": obs["image"].astype(np.uint8),
                    "goal":  rel_goal,
                    "vel":   vel,
                    "depth": info["depth"],
                    "seg":   info["seg"],
                })

            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        process_episode(
            decisions,
            np.stack(positions, axis=0),
            np.stack(actions, axis=0),
            writer,
            horizon_k=args.horizon_k,
            stride=args.stride,
            chunk_h=args.chunk_h,
            perception_size=args.perception_size,
        )
        episodes += 1
        print(f"episode {episodes:4d}  len={len(positions):4d}  "
              f"decisions={len(decisions):3d}  "
              f"total_samples={writer.total:,}/{args.steps:,}")

    writer.flush()
    env.close()

    meta = {
        "checkpoint": args.checkpoint,
        "total_samples": writer.total,
        "episodes": episodes,
        "img_size": args.img_size,
        "perception_size": args.perception_size,
        "perception_stride": args.perception_stride,
        "horizon_k": args.horizon_k,
        "stride": args.stride,
        "horizon_seconds": args.horizon_k * args.stride * 0.1,
        "chunk_h": args.chunk_h,
        "control_hz": 10,
        "waypoints_file": args.waypoints_file,
        "teacher": "blind_oracle",
        "note": "pipeline-validation data; not true obstacle avoidance",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. {writer.total:,} samples across {episodes} episodes.")
    print(f"Meta → {os.path.join(args.out, 'meta.json')}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect oracle demos for the dual-system pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained oracle PPO .zip checkpoint")
    parser.add_argument("--steps", type=int, default=20_000,
                        help="Number of DECISION-POINT samples to collect (each one "
                             "has an image+depth+seg; with the default perception "
                             "stride of 5, 20,000 samples = ~100,000 control ticks)")
    parser.add_argument("--out", type=str, default="data/demos",
                        help="Output directory for dataset shards")
    parser.add_argument("--waypoints-file", type=str, default="waypoints.json",
                        help="Waypoints JSON (same as Stage 3 training)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Torch device for the oracle policy")

    # Recording schema
    parser.add_argument("--img-size", type=int, default=224,
                        help="Stored scene image resolution (SigLIP wants 224)")
    parser.add_argument("--perception-size", type=int, default=64,
                        help="Stored depth/segmentation resolution")
    parser.add_argument("--perception-stride", type=int, default=5,
                        help="Fetch image+depth+seg every N control ticks (default 5, "
                             "about 2Hz, matching System 2's intended decision rate). "
                             "Keeps the velocity control loop smooth and avoids the "
                             "AirSim memory growth seen from per-tick image RPCs.")
    parser.add_argument("--horizon-k", type=int, default=6,
                        help="Number of future trajectory waypoints")
    parser.add_argument("--stride", type=int, default=5,
                        help="Steps between trajectory waypoints (5 = 0.5s @ 10Hz)")
    parser.add_argument("--chunk-h", type=int, default=16,
                        help="Action-chunk length for System 1")
    parser.add_argument("--shard-size", type=int, default=2000,
                        help="Samples per .npz shard")

    # Episode randomization (mirror Stage 3)
    parser.add_argument("--goal-min-dist", type=float, default=5.0)
    parser.add_argument("--goal-max-dist", type=float, default=20.0)
    parser.add_argument("--start-radius", type=float, default=8.0)
    parser.add_argument("--max-ep-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    collect(args)
