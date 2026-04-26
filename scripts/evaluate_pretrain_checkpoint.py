"""
Evaluate the pretrained CNN warm-start checkpoint.

This script does not train. It loads cnn_pretrained.pth, runs frozen ISP+CNN on
the validation split, reports L1_Y/L1_UV/VIF/NRQM/UNIQUE, compares them against
baseline, and saves a JSON report.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isp.config.config_reader import read_config
from isp.models.residual_cnn import ResidualCNN
from scripts.run_pretrain_cnn import BASELINE, run_per_scene_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained CNN")
    parser.add_argument("--ckpt", default="artifacts/checkpoints/cnn_pretrained.pth")
    parser.add_argument("--config", default="data/imx623.toml")
    parser.add_argument("--splits-json", default="dataset/splits_mini.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--eval-max-frames", type=int, default=3)
    parser.add_argument(
        "--baseline-mode",
        choices=["same-split", "static"],
        default="same-split",
        help="same-split evaluates ISP-only on the same frames; "
        "static uses the constants from run_pretrain_cnn.py",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/checkpoints/pretrain_eval_metrics.json")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_cnn(ckpt_path: Path, device: str) -> tuple[ResidualCNN, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    config = ckpt.get("config", {})

    cnn = ResidualCNN(
        in_channels=3,
        hidden_channels=int(config.get("hidden_ch", 32)),
        out_channels=3,
        num_blocks=int(config.get("num_blocks", 5)),
        num_groups=int(config.get("num_groups", 8)),
    ).to(device)
    cnn.load_state_dict(state)
    cnn.eval()
    return cnn, ckpt


def average_baseline(scenes: list[str]) -> dict:
    return {
        metric: sum(BASELINE[scene][metric] for scene in scenes) / len(scenes)
        for metric in ["vif", "nrqm", "unique", "l1_y", "l1_uv"]
    }


def average_results(results: dict) -> dict:
    scenes = list(results.keys())
    return {
        metric: sum(float(results[scene][metric]) for scene in scenes) / len(scenes)
        for metric in ["vif", "nrqm", "unique", "l1_y", "l1_uv"]
    }


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    ckpt_path = resolve(args.ckpt)
    config_path = resolve(args.config)
    splits_json = resolve(args.splits_json)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = read_config(str(config_path), device=device)
    cnn, ckpt = load_cnn(ckpt_path, device)

    print(f"Checkpoint: {ckpt_path}")
    print(f"  epoch:    {ckpt.get('epoch', 'unknown')}")
    print(f"  val_loss: {ckpt.get('val_loss', 'unknown')}")
    print(f"Evaluating split={args.split}, max_frames={args.eval_max_frames}/scene")

    results = run_per_scene_eval(
        cnn=cnn,
        config=config,
        config_path=config_path,
        splits_json=splits_json,
        split_name=args.split,
        device=device,
        max_frames=args.eval_max_frames,
    )

    scenes = list(results.keys())
    if args.baseline_mode == "same-split":
        print("\nEvaluating ISP-only baseline on the same split/frames...")
        baseline_results = run_per_scene_eval(
            cnn=None,
            config=config,
            config_path=config_path,
            splits_json=splits_json,
            split_name=args.split,
            device=device,
            max_frames=args.eval_max_frames,
        )
        baseline_avg = average_results(baseline_results)
    else:
        baseline_results = {scene: dict(BASELINE[scene]) for scene in scenes}
        baseline_avg = average_baseline(scenes)

    result_avg = average_results(results)

    checks = {
        "checkpoint_exists": ckpt_path.exists(),
        "vif_above_baseline_avg": result_avg["vif"] > baseline_avg["vif"],
        "l1_y_better_than_baseline_avg": result_avg["l1_y"] < baseline_avg["l1_y"],
        "l1_uv_better_than_baseline_avg": result_avg["l1_uv"] < baseline_avg["l1_uv"],
        "l1_y_or_l1_uv_better_than_baseline_avg": (
            result_avg["l1_y"] < baseline_avg["l1_y"] or result_avg["l1_uv"] < baseline_avg["l1_uv"]
        ),
    }

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "split": args.split,
        "eval_max_frames_per_scene": args.eval_max_frames,
        "baseline_mode": args.baseline_mode,
        "per_scene": results,
        "baseline_per_scene": baseline_results,
        "average": result_avg,
        "baseline_average": baseline_avg,
        "checks": checks,
    }

    print("\nScene metrics:")
    for scene in scenes:
        r = results[scene]
        b = baseline_results[scene]
        print(
            f"  {scene:<8s} "
            f"VIF {r['vif']:.4f} ({r['vif'] - b['vif']:+.4f})  "
            f"NRQM {r['nrqm']:.4f} ({r['nrqm'] - b['nrqm']:+.4f})  "
            f"UNIQUE {r['unique']:.4f} ({r['unique'] - b['unique']:+.4f})  "
            f"L1_Y {r['l1_y']:.4f}  L1_UV {r['l1_uv']:.4f}"
        )

    print("\nAverage:")
    for metric in ["vif", "nrqm", "unique", "l1_y", "l1_uv"]:
        delta = result_avg[metric] - baseline_avg[metric]
        print(
            f"  {metric:<7s} {result_avg[metric]:.6f} "
            f"(baseline {baseline_avg[metric]:.6f}, delta {delta:+.6f})"
        )

    print("\nChecks:")
    for name, ok in checks.items():
        print(f"  {name:<40s}: {'OK' if ok else 'FAIL'}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
