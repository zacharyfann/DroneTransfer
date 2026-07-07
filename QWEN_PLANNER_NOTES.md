# Qwen3.5-2B LoRA Planner Notes

This repository now includes `train_planner_qwen_lora.py` for System 2
trajectory planning (`future_offsets`, ~3s horizon).

## Why this setup

- **LoRA (parameter-efficient fine-tuning):** proven to work in VLM/VLA
  fine-tuning when full-model updates are too expensive.
- **Masked Huber trajectory loss:** robust for waypoint regression with noisy
  tails and padded labels.
- **Trajectory smoothness regularizer:** common in planning stacks to reduce
  zig-zag outputs.
- **Cosine LR + warmup, grad clipping:** standard stable recipe for adapter
  fine-tuning.

## Data assumptions

Input shards from `collect_demos.py`:

- `image` (`uint8`, 224x224x3)
- `goal`, `vel`
- `future_offsets` (`Kx3`)
- `future_mask` (`K`)

Collection now supports:

- `--min-future-valid` (default `4`): drops near-episode-end samples with too
  many padded future waypoints.

## Launch example

```bash
python3 train_planner_qwen_lora.py \
  --data-dir data/demos \
  --out-dir checkpoints/planner_qwen_lora \
  --model-name Qwen/Qwen3.5-2B \
  --epochs 5 \
  --batch-size 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --horizon-k 6 \
  --horizon-dt 0.5
```

## Important design note

The Oracle PPO MLP and this planner head are different models and are trained
separately:

- Oracle PPO: RL policy (`goal+vel -> action`)
- Qwen planner: supervised trajectory model (`image+goal+vel -> future path`)

The Oracle checkpoint is used to **collect labels**, not to initialize this
planner's regression head.

