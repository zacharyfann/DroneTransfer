"""
train_planner_qwen_lora.py
--------------------------

Train a 3-second trajectory planner on demo shards using:
  - Qwen3.5-2B (multimodal backbone)
  - LoRA adapters (parameter-efficient fine-tuning)
  - A small regression head that predicts future_offsets (K x 3)

This script trains System 2 only (planner). System 1 (action head / DiT) is
separate and should be trained later on action_chunk labels.

Expected dataset format:
  Sharded .npz files from collect_demos.py with keys:
    image, goal, vel, future_offsets, future_mask
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _load_multimodal_model(model_name: str):
    # Keep this resilient to HF class renames across versions.
    from transformers import AutoProcessor

    model = None
    model_errs: list[str] = []

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as e:
        model_errs.append(f"AutoModelForImageTextToText: {e}")

    if model is None:
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            model_errs.append(f"AutoModelForVision2Seq: {e}")

    if model is None:
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            model_errs.append(f"AutoModelForCausalLM: {e}")

    if model is None:
        raise RuntimeError(
            "Failed to load multimodal model. Tried common HF loaders:\n"
            + "\n".join(model_errs)
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


class TrajectoryHead(nn.Module):
    """Predict Kx3 relative waypoints from a fused hidden vector."""

    def __init__(self, hidden_size: int, horizon_k: int, dropout: float = 0.1):
        super().__init__()
        self.horizon_k = horizon_k
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon_k * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(out.shape[0], self.horizon_k, 3)


class ShardedDemoDataset(Dataset):
    """Random-access dataset over sharded .npz files."""

    def __init__(self, shard_paths: list[str]):
        if not shard_paths:
            raise ValueError("No shard files provided.")
        self.shard_paths = shard_paths
        self.lengths: list[int] = []
        self.cum: list[int] = []
        total = 0
        for p in self.shard_paths:
            with np.load(p, mmap_mode="r") as z:
                n = int(z["image"].shape[0])
            self.lengths.append(n)
            total += n
            self.cum.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> tuple[int, int]:
        shard_idx = bisect_right(self.cum, idx)
        start = 0 if shard_idx == 0 else self.cum[shard_idx - 1]
        local_idx = idx - start
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        shard_idx, local_idx = self._locate(idx)
        p = self.shard_paths[shard_idx]
        with np.load(p, mmap_mode="r") as z:
            return {
                "image": z["image"][local_idx].astype(np.uint8),
                "goal": z["goal"][local_idx].astype(np.float32),
                "vel": z["vel"][local_idx].astype(np.float32),
                "future_offsets": z["future_offsets"][local_idx].astype(np.float32),
                "future_mask": z["future_mask"][local_idx].astype(np.float32),
            }


@dataclass
class Batch:
    images: list[Image.Image]
    prompts: list[str]
    future_offsets: torch.Tensor
    future_mask: torch.Tensor


def _make_prompt(goal: np.ndarray, vel: np.ndarray, horizon_k: int, dt_sec: float) -> str:
    # Keep prompt structured and deterministic for stable conditioning.
    return (
        "You are a drone trajectory planner.\n"
        "Given the current front camera image, relative goal (meters), and current velocity (m/s), "
        f"predict {horizon_k} future relative waypoints at {dt_sec:.2f}s intervals.\n"
        f"Goal dx dy dz: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]\n"
        f"Velocity vx vy vz: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]\n"
    )


def collate_samples(samples: list[dict[str, np.ndarray]], horizon_k: int, dt_sec: float) -> Batch:
    images: list[Image.Image] = []
    prompts: list[str] = []
    fut = []
    msk = []
    for s in samples:
        images.append(Image.fromarray(s["image"]))
        prompts.append(_make_prompt(s["goal"], s["vel"], horizon_k, dt_sec))
        fut.append(s["future_offsets"])
        msk.append(s["future_mask"])
    return Batch(
        images=images,
        prompts=prompts,
        future_offsets=torch.from_numpy(np.stack(fut, axis=0)).float(),
        future_mask=torch.from_numpy(np.stack(msk, axis=0)).float(),
    )


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: [B, K, 3], mask: [B, K]
    per_wp = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)  # [B, K]
    denom = mask.sum().clamp(min=1.0)
    return (per_wp * mask).sum() / denom


def _extract_hidden(outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    # Last non-pad token hidden state.
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        last = outputs.hidden_states[-1]
    elif hasattr(outputs, "last_hidden_state"):
        last = outputs.last_hidden_state
    else:
        raise RuntimeError("Model output has no hidden states.")
    idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    return last[torch.arange(last.shape[0], device=last.device), idx]


def compute_path_metrics(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    msk: torch.Tensor,
) -> dict[str, float]:
    # ADE: average L2 over valid waypoints.
    l2 = torch.norm(pred - tgt, dim=-1)  # [B, K]
    ade = (l2 * msk).sum() / msk.sum().clamp(min=1.0)

    # FDE: final valid waypoint L2.
    b, k = msk.shape
    fde_vals = []
    for i in range(b):
        valid = torch.nonzero(msk[i] > 0.5, as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        last = int(valid[-1].item())
        fde_vals.append(l2[i, last])
    fde = torch.stack(fde_vals).mean() if fde_vals else torch.tensor(0.0, device=pred.device)
    return {"ade": float(ade.item()), "fde": float(fde.item())}


def run_epoch(
    *,
    model: nn.Module,
    processor: Any,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    smoothness_coef: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    head.train(training)

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0

    for batch in loader:
        b: Batch = batch
        pixel_inputs = processor(
            images=b.images,
            text=b.prompts,
            return_tensors="pt",
            padding=True,
        )
        pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
        future_offsets = b.future_offsets.to(device)
        future_mask = b.future_mask.to(device)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**pixel_inputs, output_hidden_states=True, use_cache=False)
                fused = _extract_hidden(outputs, pixel_inputs["attention_mask"])
                pred = head(fused)

                loss_main = masked_huber(pred, future_offsets, future_mask)

                # Small curvature regularizer to reduce zig-zag plans.
                if pred.shape[1] >= 3 and smoothness_coef > 0:
                    acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
                    loss_smooth = acc.abs().mean()
                else:
                    loss_smooth = torch.tensor(0.0, device=device)

                loss = loss_main + smoothness_coef * loss_smooth

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
                    optimizer.step()

        metrics = compute_path_metrics(pred.detach(), future_offsets, future_mask)
        total_loss += float(loss.detach().item())
        total_ade += metrics["ade"]
        total_fde += metrics["fde"]
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "ade": 0.0, "fde": 0.0}
    return {
        "loss": total_loss / n_batches,
        "ade": total_ade / n_batches,
        "fde": total_fde / n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B LoRA trajectory planner")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing collect_demos shard_*.npz files")
    parser.add_argument("--out-dir", type=str, default="checkpoints/planner_qwen_lora",
                        help="Output directory for adapters/head/checkpoints")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-2B",
                        help="HF model id for multimodal Qwen backbone")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-k", type=int, default=6)
    parser.add_argument("--horizon-dt", type=float, default=0.5,
                        help="Seconds between waypoints (default 0.5s)")
    parser.add_argument("--smoothness-coef", type=float, default=0.01,
                        help="Curvature regularization weight")
    parser.add_argument("--dropout", type=float, default=0.1)

    # LoRA controls.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
                        help="Comma-separated module names for LoRA")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz found in {args.data_dir}")
    n_val = max(1, int(len(shard_paths) * args.val_split))
    val_paths = shard_paths[-n_val:]
    train_paths = shard_paths[:-n_val] if len(shard_paths) > 1 else shard_paths

    train_ds = ShardedDemoDataset(train_paths)
    val_ds = ShardedDemoDataset(val_paths)

    def _collate_fn(items):
        return collate_samples(items, horizon_k=args.horizon_k, dt_sec=args.horizon_dt)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    model, processor = _load_multimodal_model(args.model_name)
    hidden_size = int(getattr(model.config, "hidden_size", 2048))

    from peft import LoraConfig, get_peft_model
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    head = TrajectoryHead(hidden_size=hidden_size, horizon_k=args.horizon_k, dropout=args.dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    # Train LoRA params + regression head only.
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_ade = float("inf")
    global_step = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            processor=processor,
            head=head,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            smoothness_coef=args.smoothness_coef,
        )
        scheduler.step()
        global_step += len(train_loader)

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                processor=processor,
                head=head,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                smoothness_coef=args.smoothness_coef,
            )

        row = {
            "epoch": float(epoch),
            "step": float(global_step),
            "train_loss": train_metrics["loss"],
            "train_ade": train_metrics["ade"],
            "train_fde": train_metrics["fde"],
            "val_loss": val_metrics["loss"],
            "val_ade": val_metrics["ade"],
            "val_fde": val_metrics["fde"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[epoch {epoch:02d}] "
            f"train loss={train_metrics['loss']:.4f} ade={train_metrics['ade']:.3f} fde={train_metrics['fde']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} ade={val_metrics['ade']:.3f} fde={val_metrics['fde']:.3f}"
        )

        # Save last
        ckpt_dir = os.path.join(args.out_dir, f"epoch_{epoch:03d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(os.path.join(ckpt_dir, "adapter"))
        processor.save_pretrained(os.path.join(ckpt_dir, "adapter"))
        torch.save(
            {
                "head_state_dict": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            },
            os.path.join(ckpt_dir, "planner_head.pt"),
        )

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            best_dir = os.path.join(args.out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(os.path.join(best_dir, "adapter"))
            processor.save_pretrained(os.path.join(best_dir, "adapter"))
            torch.save(
                {"head_state_dict": head.state_dict(), "val_metrics": val_metrics, "args": vars(args)},
                os.path.join(best_dir, "planner_head.pt"),
            )

    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val ADE: {best_val_ade:.4f} m")
    print(f"Checkpoints saved in: {args.out_dir}")


if __name__ == "__main__":
    main()

