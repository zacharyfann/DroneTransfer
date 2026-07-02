"""
train_ppo.py — Train a PPO agent on ColosseumNavEnv.

Two modes (select with --mode):

  oracle   (default)
      Obs = [relative goal (3), velocity (3)].
      No image, no VLM.  Fastest convergence; validates reward shaping first.

  vision
      Obs = image (84×84×3).  Uses SB3 CnnPolicy (NatureCNN backbone).
      Good CNN ablation before adding VLM.

  vlm
      Frozen SigLIP encodes the image to a 512-d embedding, which is
      concatenated with the relative goal and passed to an MLP actor-critic.
      Requires `pip install transformers torch Pillow`.

Usage:
  python train_ppo.py                          # oracle mode, 200k steps
  python train_ppo.py --mode vision            # CNN policy
  python train_ppo.py --mode vlm               # frozen SigLIP + MLP
  python train_ppo.py --mode oracle --steps 500000 --goal 30 0 -15
  python train_ppo.py --mode oracle --steps 100000 --checkpoint checkpoints/oracle/ppo_oracle_30000_steps

Stage 2 (smooth + random goals, resume from Stage 1 checkpoint):
  python train_ppo.py --mode oracle --randomize-goal --smooth-coef 0.03 `
      --steps <current_steps+500000> `
      --checkpoint checkpoints/oracle/ppo_oracle_final

Stage 3 (random start + longer range, resume from Stage 2 checkpoint):
  python train_ppo.py --mode oracle --randomize-goal --randomize-start `
      --smooth-coef 0.05 --action-smooth-alpha 0.6 `
      --goal-max-dist 50 --goal-alt-max 25 --start-radius 10 `
      --max-ep-steps 800 `
      --steps <current_steps+200000> `
      --checkpoint checkpoints/oracle/ppo_oracle_<last_stage2_step>_steps
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from colosseum_nav_env import ColosseumNavEnv


# ─────────────────────────────────────────────────────────────────────────────
# 0. Episode stats callback  (success / collision / timeout rates → TensorBoard)
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """Logs success/collision/timeout/stuck rates to TensorBoard.

    Reads ``dist_to_goal``, ``collision``, and ``stuck`` from the info dict
    that ColosseumNavEnv writes on every terminal step.
    """

    def __init__(self, goal_radius: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.goal_radius = goal_radius
        self._successes: list[int] = []
        self._collisions: list[int] = []
        self._timeouts: list[int] = []
        self._stucks: list[int] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            dist = info.get("dist_to_goal", float("inf"))
            collision = info.get("collision", False)
            stuck = info.get("stuck", False)

            if dist < self.goal_radius:
                self._successes.append(1)
                self._collisions.append(0)
                self._timeouts.append(0)
                self._stucks.append(0)
            elif stuck:
                self._successes.append(0)
                self._collisions.append(0)
                self._timeouts.append(0)
                self._stucks.append(1)
            elif collision:
                self._successes.append(0)
                self._collisions.append(1)
                self._timeouts.append(0)
                self._stucks.append(0)
            else:
                self._successes.append(0)
                self._collisions.append(0)
                self._timeouts.append(1)
                self._stucks.append(0)

            # Log a rolling window of the last 100 episodes
            window = 100
            n = min(len(self._successes), window)
            self.logger.record("episode/success_rate",   sum(self._successes[-n:])  / n)
            self.logger.record("episode/collision_rate", sum(self._collisions[-n:]) / n)
            self.logger.record("episode/timeout_rate",   sum(self._timeouts[-n:])   / n)
            self.logger.record("episode/stuck_rate",     sum(self._stucks[-n:])     / n)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. Oracle wrapper  (image stripped out; obs = [goal, vel])
# ─────────────────────────────────────────────────────────────────────────────

class OracleNavWrapper(gym.ObservationWrapper):
    """Drops the image; exposes only goal + velocity as a flat vector."""

    def __init__(self, env: ColosseumNavEnv) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-500.0, high=500.0,
            shape=(6,),       # [dx, dy, dz, vx, vy, vz]
            dtype=np.float32,
        )

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate([obs["goal"], obs["vel"]], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image-only wrapper (goal + vel dropped; SB3 CnnPolicy handles image)
# ─────────────────────────────────────────────────────────────────────────────

class ImageNavWrapper(gym.ObservationWrapper):
    """Returns only the image (H, W, 3) for CnnPolicy."""

    def __init__(self, env: ColosseumNavEnv) -> None:
        super().__init__(env)
        self.observation_space = env.observation_space["image"]

    def observation(self, obs: dict) -> np.ndarray:
        return obs["image"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. VLM feature extractor  (frozen SigLIP image encoder + goal concat)
# ─────────────────────────────────────────────────────────────────────────────

class FrozenSigLIPExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor that:
      1. Runs the image through a frozen SigLIP vision encoder → 512-d
      2. Concatenates the relative goal (3-d) and velocity (3-d)
      3. Projects to features_dim via a small trainable MLP

    The SigLIP weights are never updated.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        model_name: str = "google/siglip-base-patch16-224",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        from transformers import SiglipVisionModel, SiglipProcessor  # lazy import

        self._processor = SiglipProcessor.from_pretrained(model_name)
        vision_model = SiglipVisionModel.from_pretrained(model_name)

        # Freeze every VLM parameter
        for param in vision_model.parameters():
            param.requires_grad_(False)
        vision_model.eval()

        self._vision_model = vision_model
        self._img_emb_dim = vision_model.config.hidden_size   # 768 for base

        # Trainable projection: [img_emb + goal(3) + vel(3)] → features_dim
        in_dim = self._img_emb_dim + 3 + 3
        self._projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        imgs   = observations["image"]    # (B, H, W, 3) uint8
        goals  = observations["goal"]     # (B, 3) float32
        vels   = observations["vel"]      # (B, 3) float32

        # SigLIP expects float [0,1] in CHW format
        imgs_float = imgs.float() / 255.0                  # (B, H, W, 3)
        imgs_chw   = imgs_float.permute(0, 3, 1, 2)        # (B, 3, H, W)

        # Resize to model's expected input via interpolation
        imgs_resized = nn.functional.interpolate(
            imgs_chw, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Run frozen encoder (no grad)
        with torch.no_grad():
            outputs = self._vision_model(pixel_values=imgs_resized)

        # Use pooled output [CLS] or mean-pool patch tokens
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            img_emb = outputs.pooler_output                  # (B, hidden_size)
        else:
            img_emb = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)

        combined = torch.cat([img_emb, goals.float(), vels.float()], dim=-1)
        return self._projection(combined)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Environment factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    mode: str,
    goal: tuple[float, float, float],
    max_steps: int,
    smooth_coef: float = 0.0,
    action_smooth_alpha: float = 0.6,
    randomize_goal: bool = False,
    goal_min_dist: float = 10.0,
    goal_max_dist: float = 30.0,
    goal_alt_min: float = 5.0,
    goal_alt_max: float = 20.0,
    randomize_start: bool = False,
    start_radius: float = 0.0,
    stuck_patience: int = 10,
    stuck_pen: float = -20.0,
    waypoints: list | None = None,
) -> gym.Env:
    base = ColosseumNavEnv(
        goal=goal,
        img_h=84,
        img_w=84,
        max_vel=5.0,
        goal_radius=2.0,
        max_steps=max_steps,
        step_duration=0.1,
        smooth_coef=smooth_coef,
        action_smooth_alpha=action_smooth_alpha,
        randomize_goal=randomize_goal,
        goal_min_dist=goal_min_dist,
        goal_max_dist=goal_max_dist,
        goal_alt_min=goal_alt_min,
        goal_alt_max=goal_alt_max,
        randomize_start=randomize_start,
        start_radius=start_radius,
        stuck_patience=stuck_patience,
        stuck_pen=stuck_pen,
        waypoints=waypoints,
        include_image=(mode != "oracle"),  # skip camera RPC in oracle mode
    )
    if mode == "oracle":
        env = OracleNavWrapper(base)
    elif mode == "vision":
        env = ImageNavWrapper(base)
    else:   # vlm
        env = base  # full Dict obs — extractor handles decomposition
    return Monitor(env)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PPO policy configs per mode
# ─────────────────────────────────────────────────────────────────────────────

def build_ppo(
    mode: str,
    env: Any,
    log_dir: str,
    device: str = "auto",
) -> PPO:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    shared_kwargs: dict[str, Any] = dict(
        env=env,
        device=device,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb"),
        n_steps=2048,
        batch_size=256,   # larger batches dilute stuck-episode noise (was 64)
        n_epochs=7,       # fewer passes per rollout reduces overfit on bad batches (was 10)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,  # more conservative policy updates (was 0.2)
        ent_coef=0.005,
        learning_rate=3e-4,
        max_grad_norm=0.3,
    )

    if mode == "oracle":
        return PPO(
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=[256, 256]),
            **shared_kwargs,
        )

    if mode == "vision":
        return PPO(
            policy="CnnPolicy",
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[256, 256],
            ),
            **shared_kwargs,
        )

    # vlm
    return PPO(
        policy="MultiInputPolicy",
        policy_kwargs=dict(
            features_extractor_class=FrozenSigLIPExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256],
        ),
        **shared_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    goal = tuple(args.goal)
    log_dir = os.path.join("logs", args.mode)
    ckpt_dir = os.path.join("checkpoints", args.mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    waypoints = None
    if args.waypoints_file:
        with open(args.waypoints_file) as f:
            waypoints = json.load(f)
        print(f"Waypoints    : {len(waypoints)} loaded from {args.waypoints_file}")

    print(f"\nMode         : {args.mode}")
    if args.randomize_goal:
        print(f"Goal         : RANDOM  h=[{args.goal_min_dist}, {args.goal_max_dist}]m  "
              f"alt=[{args.goal_alt_min}, {args.goal_alt_max}]m")
    else:
        print(f"Goal         : {goal}")
    if args.randomize_start:
        print(f"Start        : RANDOM  radius={args.start_radius}m")
    else:
        print(f"Start        : fixed spawn")
    print(f"Smooth coef  : {args.smooth_coef}  (EMA alpha={args.action_smooth_alpha})")
    print(f"Steps        : {args.steps:,}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu" and torch.cuda.is_available():
        print("  (pass --device cuda to use the GPU for the policy network)")
    print()

    _wps = waypoints  # capture for lambda closure
    env = DummyVecEnv([lambda: make_env(
        args.mode, goal, args.max_ep_steps,
        smooth_coef=args.smooth_coef,
        action_smooth_alpha=args.action_smooth_alpha,
        randomize_goal=args.randomize_goal,
        goal_min_dist=args.goal_min_dist,
        goal_max_dist=args.goal_max_dist,
        goal_alt_min=args.goal_alt_min,
        goal_alt_max=args.goal_alt_max,
        randomize_start=args.randomize_start,
        start_radius=args.start_radius,
        stuck_patience=args.stuck_patience,
        stuck_pen=args.stuck_pen,
        waypoints=_wps,
    )])

    if args.checkpoint:
        ckpt_path = args.checkpoint.removesuffix(".zip")
        print(f"Resuming from: {ckpt_path}.zip")
        model = PPO.load(ckpt_path, env=env, device=device)
        # Override hyperparameters that may differ from the saved checkpoint
        model.max_grad_norm = 0.3
        model.batch_size = 256
        model.n_epochs = 7
        model.clip_range = lambda _: 0.15
        model.set_logger(configure(os.path.join(log_dir, "tb"), ["stdout", "tensorboard"]))
        done = model.num_timesteps
        remaining = max(args.steps - done, 0)
        print(f"  Timesteps done: {done:,}  target: {args.steps:,}  remaining: {remaining:,}")
        if remaining == 0:
            print("Already at or past --steps; nothing to train.")
            env.close()
            return
    else:
        model = build_ppo(args.mode, env, log_dir, device=device)
        remaining = args.steps

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.ckpt_freq, 2048),
            save_path=ckpt_dir,
            name_prefix=f"ppo_{args.mode}",
        ),
        EpisodeStatsCallback(goal_radius=2.0),
    ]

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(args.checkpoint),
    )

    # Name the final checkpoint with its step count so resuming never overwrites
    # a previous stage's final save (e.g. stage1 final stays as _final_70000).
    final_tag = f"final_{model.num_timesteps}"
    final_path = os.path.join(ckpt_dir, f"ppo_{args.mode}_{final_tag}")
    model.save(final_path)
    print(f"\nModel saved → {final_path}.zip")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ColosseumNavEnv")
    parser.add_argument(
        "--mode", choices=["oracle", "vision", "vlm"], default="oracle",
        help="oracle=pose-only, vision=CNN, vlm=frozen SigLIP",
    )
    parser.add_argument(
        "--steps", type=int, default=200_000,
        help="Target total timesteps (includes steps already in --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved .zip checkpoint to resume from",
    )
    parser.add_argument(
        "--goal", nargs=3, type=float, default=[20.0, 0.0, -10.0],
        metavar=("X", "Y", "Z"),
        help="Target position in NED metres",
    )
    parser.add_argument(
        "--max-ep-steps", type=int, default=500,
        help="Max steps before episode truncates",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for PPO policy (sim still runs on CPU)",
    )
    parser.add_argument(
        "--ckpt-freq", type=int, default=10_000,
        help="Save a checkpoint every N timesteps (default: 10,000)",
    )

    # ── Stage 2: smoothness + goal randomization ──────────────────────────────
    parser.add_argument(
        "--smooth-coef", type=float, default=0.0,
        help="Reward penalty weight for raw action change between steps (0=disabled). "
             "Stage 2 recommended: 0.03",
    )
    parser.add_argument(
        "--action-smooth-alpha", type=float, default=0.6,
        metavar="A",
        help="EMA alpha for velocity commands sent to sim (0=frozen, 1=no filter). "
             "Default 0.6 — lower values = smoother but more sluggish.",
    )
    parser.add_argument(
        "--randomize-goal", action="store_true",
        help="Sample a new random goal each episode (relative to start position)",
    )
    parser.add_argument(
        "--goal-min-dist", type=float, default=10.0,
        metavar="M",
        help="Min horizontal distance for randomized goals (metres)",
    )
    parser.add_argument(
        "--goal-max-dist", type=float, default=30.0,
        metavar="M",
        help="Max horizontal distance for randomized goals (metres)",
    )
    parser.add_argument(
        "--goal-alt-min", type=float, default=5.0,
        metavar="M",
        help="Min altitude for randomized goals (metres above ground)",
    )
    parser.add_argument(
        "--goal-alt-max", type=float, default=20.0,
        metavar="M",
        help="Max altitude for randomized goals (metres above ground)",
    )

    # ── Stage 3: random start position + extended range ───────────────────────
    parser.add_argument(
        "--randomize-start", action="store_true",
        help="Teleport drone to a random XY offset after each takeoff",
    )
    parser.add_argument(
        "--start-radius", type=float, default=0.0,
        metavar="M",
        help="Max horizontal offset from spawn for random start (metres). "
             "Stage 3 recommended: 10",
    )

    # ── Waypoint map (Stage 3b — obstacle-aware curriculum) ───────────────────
    parser.add_argument(
        "--waypoints-file", type=str, default=None,
        metavar="PATH",
        help="JSON file with fixed map waypoints: [[x,y,z], ...] in NED metres. "
             "When provided with --randomize-goal, goals are drawn from this list "
             "instead of random directions. Edit waypoints.json to match your map.",
    )

    # ── Stuck detection (motor bug-out / physics instability) ─────────────────
    parser.add_argument(
        "--stuck-patience", type=int, default=10,
        metavar="N",
        help="Consecutive steps of no motion before episode is terminated as stuck "
             "(default: 10 = 1 second). Lower = faster recovery, more false positives.",
    )
    parser.add_argument(
        "--stuck-pen", type=float, default=-20.0,
        metavar="R",
        help="Reward penalty applied when a stuck termination is triggered "
             "(default: -20, less severe than collision penalty of -50).",
    )

    args = parser.parse_args()
    train(args)
