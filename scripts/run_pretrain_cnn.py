"""
Pretrain CNN on frozen ISP.

Trains ResidualCNN for 20 epochs with ISP parameters frozen.
Every eval_every epochs runs full-frame evaluation (VIF, NRQM, UNIQUE).
Saves best checkpoint to artifacts/checkpoints/cnn_pretrained.pth.

Supports two modes:
  - Fast mode: uses precomputed ISP outputs from HDF5 (y_isp, uv_isp).
    Run scripts/precompute_isp_outputs.py first.
  - Slow mode: runs ISP per patch on every batch (fallback if no precomputed data).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from isp.color.conversions import yuv420_to_yuv444, yuv444_to_yuv420
from isp.config.config_reader import read_config
from isp.data.dataset_utils import ISPDataset, create_dataloader
from isp.evaluation.evaluation_utils import (
    evaluate,
    limit_eval_items,
    load_split_items,
)
from isp.models.residual_cnn import ResidualCNN, count_trainable_parameters
from isp.pipeline.pipeline import ISPPipeline
from isp.training.training_utils import train_step

ISP_PARAMS = {
    "day": dict(
        denoise_eps=1e-12,
        ltm_a=0.5,
        ltm_detail_gain=30,
        ltm_detail_threshold=0.35,
        hist_target_mean=0.445,
        hist_target_std=0.162,
        post_denoise_radius=4,
        post_denoise_eps=0.001,
        raw_y_full_blend=0.4,
        sharp_amount=0.3,
        saturation=1.2,
    ),
    "night": dict(
        denoise_eps=1e-12, ltm_a=0.3, ltm_detail_gain=8, ltm_detail_threshold=0.4, sharp_amount=0.8
    ),
    "tunnel": {},
}

BASELINE = {
    "day": {
        "vif": 0.702358,
        "nrqm": 5.227967,
        "unique": 0.124453,
        "l1_y": 0.048739,
        "l1_uv": 0.023350,
    },
    "night": {
        "vif": 0.519090,
        "nrqm": 7.075381,
        "unique": 0.135025,
        "l1_y": 0.058578,
        "l1_uv": 0.010266,
    },
    "tunnel": {
        "vif": 0.693219,
        "nrqm": 6.870870,
        "unique": 0.076290,
        "l1_y": 0.267548,
        "l1_uv": 0.021035,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain CNN (ISP frozen)")
    p.add_argument("--train-h5", default="dataset/train_patches.h5")
    p.add_argument("--splits-json", default="dataset/splits.json")
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--lambda-uv", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5, help="Full evaluation every N epochs")
    p.add_argument(
        "--eval-max-frames",
        type=int,
        default=3,
        help="Max frames per scene during eval (for speed)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping: stop after N evals without improvement",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use mixed-precision (AMP) training (default: True)",
    )
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--cache-ram",
        action="store_true",
        help="For compact/precomputed HDF5, load tensors into RAM once "
        "to avoid slow compressed random I/O in Colab",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="artifacts/checkpoints")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def seed_everything(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_isp(isp):
    """Freeze all ISP parameters so only CNN trains."""
    frozen = 0
    for _name, param in isp.named_parameters():
        param.requires_grad = False
        frozen += 1
    print(f"ISP frozen: {frozen} parameters set to requires_grad=False")


class FastInMemoryDataset(Dataset):
    """Compact HDF5 dataset cached in CPU RAM for fast Colab pretraining."""

    def __init__(self, h5_path: str):
        t0 = time.time()
        with h5py.File(h5_path, "r") as h5_file:
            missing = [
                name for name in ("y_isp", "uv_isp", "y_ref", "uv_ref") if name not in h5_file
            ]
            if missing:
                raise KeyError(f"Cannot cache {h5_path}: missing datasets {missing}")

            self.y_isp = torch.from_numpy(h5_file["y_isp"][:].copy())
            self.uv_isp = torch.from_numpy(h5_file["uv_isp"][:].copy())
            self.y_ref = torch.from_numpy(h5_file["y_ref"][:].copy())
            self.uv_ref = torch.from_numpy(h5_file["uv_ref"][:].copy())

        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in [self.y_isp, self.uv_isp, self.y_ref, self.uv_ref]
        )
        print(
            f"Cached compact HDF5 in RAM: {len(self):,} patches, "
            f"{total_bytes / 1024**3:.2f} GiB, {time.time() - t0:.1f}s"
        )

    def __len__(self):
        return int(self.y_isp.shape[0])

    def __getitem__(self, index: int):
        return {
            "y_isp": self.y_isp[index],
            "uv_isp": self.uv_isp[index],
            "y_ref": self.y_ref[index].float().unsqueeze(0) / 255.0,
            "uv_ref": self.uv_ref[index].float() / 255.0,
        }


def fast_train_step(cnn, optimizer, batch, device, lambda_uv, scaler=None, amp_enabled=False):
    """Train step using precomputed ISP outputs — no ISP forward needed."""
    y_isp = batch["y_isp"].to(device).float()
    uv_isp = batch["uv_isp"].to(device).float()
    y_ref = batch["y_ref"].to(device)
    uv_ref = batch["uv_ref"].to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        yuv444_isp = yuv420_to_yuv444(y_isp, uv_isp)
        residual = cnn(yuv444_isp)
        yuv444_pred = torch.clamp(yuv444_isp + residual, 0.0, 1.0)
        y_pred, uv_pred = yuv444_to_yuv420(yuv444_pred)

        l1_y = F.l1_loss(y_pred.float(), y_ref)
        l1_uv = F.l1_loss(uv_pred.float(), uv_ref)
        loss = l1_y + lambda_uv * l1_uv

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {
        "loss": float(loss.item()),
        "l1_y": float(l1_y.item()),
        "l1_uv": float(l1_uv.item()),
    }


def run_per_scene_eval(cnn, config, config_path, splits_json, split_name, device, max_frames):
    """Evaluate ISP+CNN per scene, return per-scene results."""
    all_results = {}

    for scene_name, scene_params in ISP_PARAMS.items():
        isp_eval = ISPPipeline(config, device=device, **scene_params)
        isp_eval.eval()

        all_items = load_split_items(str(splits_json), split_name)
        scene_items = [item for item in all_items if item["scene"] == scene_name]

        if not scene_items:
            continue

        scene_items = limit_eval_items(scene_items, max_frames)

        result = evaluate(
            isp=isp_eval,
            model=cnn,
            eval_items=scene_items,
            config_path=str(config_path),
            device=device,
            compute_iqa=True,
            verbose=False,
        )
        all_results[scene_name] = result

        del isp_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def print_eval_results(epoch, results, baseline):
    """Print per-scene eval results with baseline comparison."""
    del epoch
    print(
        f"\n  {'Scene':<8s} {'VIF':>8s} {'dVIF':>8s} {'NRQM':>8s} {'dNRQM':>8s} "
        f"{'UNIQUE':>8s} {'dUNIQ':>8s} {'L1_Y':>8s} {'L1_UV':>8s}"
    )
    print(f"  {'-' * 80}")

    for scene in ["day", "night", "tunnel"]:
        if scene not in results:
            continue
        r = results[scene]
        b = baseline[scene]
        d_vif = r["vif"] - b["vif"]
        d_nrqm = r["nrqm"] - b["nrqm"]
        d_uniq = r["unique"] - b["unique"]
        print(
            f"  {scene:<8s} {r['vif']:8.4f} {d_vif:+8.4f} {r['nrqm']:8.4f} {d_nrqm:+8.4f} "
            f"{r['unique']:8.4f} {d_uniq:+8.4f} {r['l1_y']:8.4f} {r['l1_uv']:8.4f}"
        )


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    seed_everything(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    amp_enabled = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    print(f"Seed: {args.seed}, AMP: {amp_enabled}, cudnn.benchmark: {device == 'cuda'}")

    config_path = resolve(args.config)
    config = read_config(str(config_path), device=device)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    train_h5 = resolve(args.train_h5)
    splits_json = resolve(args.splits_json)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_probe = ISPDataset(str(train_h5), return_meta=False)
    use_fast_path = dataset_probe.has_isp_outputs
    dataset_probe.close()

    if use_fast_path:
        print("*** FAST MODE: using precomputed ISP outputs (y_isp, uv_isp)")
    else:
        print("*** SLOW MODE: running ISP per batch (consider precompute_isp_outputs.py)")

    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    n_params = count_trainable_parameters(cnn)
    print(f"CNN trainable params: {n_params:,}")

    optimizer = Adam([p for p in cnn.parameters() if p.requires_grad], lr=args.lr)

    isp = ISPPipeline(config, device=device, **ISP_PARAMS["day"])
    freeze_isp(isp)
    isp.eval()

    if args.cache_ram:
        if not use_fast_path:
            raise ValueError("--cache-ram requires precomputed y_isp/uv_isp in HDF5")
        if args.num_workers != 0:
            print("cache-ram is enabled; using num_workers=0 to avoid RAM duplication")
        train_dataset = FastInMemoryDataset(str(train_h5))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
    else:
        train_loader = create_dataloader(
            h5_path=str(train_h5),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            return_meta=False,
        )
    print(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} patches)")

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    print(f"\nStarting pretraining: {args.epochs} epochs, lr={args.lr}, lambda_uv={args.lambda_uv}")
    print(
        f"Eval every {args.eval_every} epochs on val split "
        f"(max {args.eval_max_frames} frames/scene)"
    )
    print(f"Early stopping patience: {args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        cnn.train()
        epoch_loss = 0.0
        epoch_l1y = 0.0
        epoch_l1uv = 0.0
        n_batches = 0

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}", leave=False)

        for batch in pbar:
            if use_fast_path:
                metrics = fast_train_step(
                    cnn=cnn,
                    optimizer=optimizer,
                    batch=batch,
                    device=device,
                    lambda_uv=args.lambda_uv,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                )
            else:
                metrics = train_step(
                    isp=isp,
                    cnn=cnn,
                    optimizer=optimizer,
                    batch=batch,
                    pattern=pattern,
                    lambda_uv=args.lambda_uv,
                )

            epoch_loss += metrics["loss"]
            epoch_l1y += metrics["l1_y"]
            epoch_l1uv += metrics["l1_uv"]
            n_batches += 1

            pbar.set_postfix(loss=f"{metrics['loss']:.5f}", l1y=f"{metrics['l1_y']:.5f}")

        pbar.close()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_l1y = epoch_l1y / n_batches
        avg_l1uv = epoch_l1uv / n_batches

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_l1y": avg_l1y,
            "train_l1uv": avg_l1uv,
            "elapsed_s": elapsed,
        }

        print(
            f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.6f}  "
            f"l1_y={avg_l1y:.6f}  l1_uv={avg_l1uv:.6f}  ({elapsed:.1f}s)"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n  Running full evaluation (epoch {epoch})...")
            cnn.eval()

            eval_results = run_per_scene_eval(
                cnn=cnn,
                config=config,
                config_path=config_path,
                splits_json=splits_json,
                split_name="val",
                device=device,
                max_frames=args.eval_max_frames,
            )

            print_eval_results(epoch, eval_results, BASELINE)

            val_scenes = [r for r in eval_results.values() if r is not None]
            if val_scenes:
                val_l1y = sum(r["l1_y"] for r in val_scenes) / len(val_scenes)
                val_l1uv = sum(r["l1_uv"] for r in val_scenes) / len(val_scenes)
                val_vif = sum(r["vif"] for r in val_scenes) / len(val_scenes)
                val_nrqm = sum(r["nrqm"] for r in val_scenes) / len(val_scenes)
                val_unique = sum(r["unique"] for r in val_scenes) / len(val_scenes)
                val_loss = val_l1y + args.lambda_uv * val_l1uv

                epoch_record.update(
                    {
                        "val_loss": val_loss,
                        "val_l1y": val_l1y,
                        "val_l1uv": val_l1uv,
                        "val_vif": val_vif,
                        "val_nrqm": val_nrqm,
                        "val_unique": val_unique,
                        "val_per_scene": eval_results,
                    }
                )

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    ckpt_path = output_dir / "cnn_pretrained.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": cnn.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_vif": val_vif,
                            "val_nrqm": val_nrqm,
                            "val_unique": val_unique,
                        },
                        ckpt_path,
                    )
                    print(
                        f"\n  *** New best val_loss={val_loss:.6f} "
                        f"(VIF={val_vif:.4f}) — saved to {ckpt_path}"
                    )
                else:
                    patience_counter += 1
                    print(
                        f"\n  val_loss={val_loss:.6f} "
                        f"(best={best_val_loss:.6f} @ epoch {best_epoch})"
                        f"  patience {patience_counter}/{args.patience}"
                    )

        history.append(epoch_record)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}: no improvement for {args.patience} evals")
            break

    print(f"\n{'=' * 60}")
    print("PRETRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best epoch: {best_epoch}  val_loss={best_val_loss:.6f}")
    print(f"Checkpoint: {output_dir / 'cnn_pretrained.pth'}")

    history_path = output_dir / "pretrain_history.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 8)
        return obj

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(history), f, indent=2, ensure_ascii=False)
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
