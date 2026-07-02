"""
test_env.py — Smoke-test ColosseumNavEnv without any RL.

Runs N episodes with random actions so you can verify:
  - Colosseum connects and the drone arms / takes off
  - Observations have the right shapes and dtypes
  - Rewards, termination flags, and info dict look sensible
  - Reset brings the drone back to origin each episode

Make sure Colosseum (Blocks) is running before executing this script.
"""

import argparse
import pprint
import time

import numpy as np

from colosseum_nav_env import ColosseumNavEnv


def run_smoke_test(
    n_episodes: int = 2,
    max_steps_per_ep: int = 30,
    goal: tuple = (20.0, 0.0, -10.0),
    verbose: bool = True,
) -> None:
    env = ColosseumNavEnv(
        goal=goal,
        img_h=84,
        img_w=84,
        max_vel=3.0,
        goal_radius=2.0,
        max_steps=max_steps_per_ep,
        step_duration=0.1,
    )

    print(f"\nObservation space:\n{env.observation_space}")
    print(f"Action space:      {env.action_space}\n")

    for ep in range(n_episodes):
        print(f"{'='*50}")
        print(f"  Episode {ep + 1} / {n_episodes}")
        print(f"{'='*50}")

        obs, info = env.reset()

        print("  reset() observation shapes:")
        for k, v in obs.items():
            print(f"    {k:8s}: shape={v.shape}  dtype={v.dtype}")
        print(f"  reset() info: {info}")

        ep_reward = 0.0
        start = time.time()

        for step in range(max_steps_per_ep):
            action = env.action_space.sample()   # random action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if verbose and step % 5 == 0:
                print(
                    f"  step {step:03d} | "
                    f"dist={info['dist_to_goal']:6.2f}m | "
                    f"reward={reward:+7.3f} | "
                    f"collision={info['collision']}"
                )

            if terminated or truncated:
                reason = "SUCCESS" if info["dist_to_goal"] < env.goal_radius else \
                         "COLLISION" if info["collision"] else "TIMEOUT"
                print(f"\n  Episode ended: {reason}")
                break

        elapsed = time.time() - start
        print(f"  Total reward: {ep_reward:.2f}   ({elapsed:.1f}s)\n")

    env.close()
    print("Done — env closed cleanly.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test ColosseumNavEnv")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--goal", nargs=3, type=float,
                        default=[20.0, 0.0, -10.0],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_smoke_test(
        n_episodes=args.episodes,
        max_steps_per_ep=args.steps,
        goal=tuple(args.goal),
        verbose=not args.quiet,
    )
