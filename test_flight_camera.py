import argparse
import numpy as np
import cv2

def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one RGB frame from backend env")
    parser.add_argument("--sim-backend", choices=["airsim", "isaac"], default="airsim")
    parser.add_argument("--isaac-task", type=str, default=None)
    parser.add_argument("--isaac-vis", action="store_true")
    parser.add_argument("--out", type=str, default="camera_test.png")
    args = parser.parse_args()

    if args.sim_backend == "airsim":
        from colosseum_nav_env import ColosseumNavEnv

        env = ColosseumNavEnv(include_image=True)
    else:
        if not args.isaac_task:
            raise ValueError("--isaac-task is required when --sim-backend isaac")
        from isaac_nav_env import IsaacNavEnv

        env = IsaacNavEnv(
            task_id=args.isaac_task,
            include_image=True,
            headless=(not args.isaac_vis),
        )

    obs, _info = env.reset()
    obs, _reward, terminated, truncated, _info = env.step(
        np.array([0.5, 0.0, 0.0], dtype=np.float32)
    )
    if terminated or truncated:
        obs, _info = env.reset()
    img = obs["image"]
    cv2.imwrite(args.out, img)
    print(f"Saved {args.out}", img.shape)
    env.close()


if __name__ == "__main__":
    main()