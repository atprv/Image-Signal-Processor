"""
End-to-end training of the differentiable ISP and the residual CNN.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from isp.config.config_reader import read_config
from isp.data.dataset_utils import create_dataloader
from isp.evaluation.baseline_io import load_e2e_minival_baseline
from isp.evaluation.composite_score import (
    compute_composite,
    compute_composite_terms,
    load_composite_config,
)
from isp.evaluation.evaluation_utils import (
    evaluate,
    limit_eval_items,
    load_split_items,
)
from isp.models.residual_cnn import ResidualCNN, count_trainable_parameters
from isp.pipeline.pipeline import ISPPipeline
from isp.training.quality_loss import QualityLossWeights
from isp.training.training_utils import (
    TRAINABLE_ISP_PARAM_KEYS,
    e2e_train_step,
)

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
    ),
    "night": dict(
        denoise_eps=1e-12, ltm_a=0.3, ltm_detail_gain=8, ltm_detail_threshold=0.4, sharp_amount=0.8
    ),
    "tunnel": dict(
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
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end ISP+CNN training")
    p.add_argument("--train-h5", default="dataset/train_patches.h5")
    p.add_argument("--splits-json", default="dataset/splits.json")
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument(
        "--ckpt",
        default="artifacts/checkpoints/cnn_pretrained.pth",
        help="Pretrained CNN warm-start checkpoint.",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument(
        "--batch-size", type=int, default=4, help="E2E runs ISP per patch (slow). Use small batch."
    )
    p.add_argument("--lr-isp", type=float, default=1e-4)
    p.add_argument("--lr-cnn", type=float, default=5e-4)
    p.add_argument("--lambda-uv", type=float, default=1.0)

    p.add_argument(
        "--loss",
        choices=["proxy", "quality"],
        default="quality",
        help="proxy = L1(Y)+lambda_uv*L1(UV); quality = MS_SSIM + VIF + UNIQUE + L1(UV).",
    )
    p.add_argument("--w-ssim", type=float, default=1.0, help="MS-SSIM weight (quality loss only).")
    p.add_argument(
        "--w-vif",
        type=float,
        default=5.0,
        help="VIF weight (quality loss only). Primary push toward higher VIF.",
    )
    p.add_argument("--w-unique", type=float, default=0.3, help="UNIQUE weight (quality loss only).")
    p.add_argument(
        "--w-l1-y",
        type=float,
        default=0.0,
        help="Y L1 reference-preserving weight (quality loss only).",
    )
    p.add_argument(
        "--w-uv", type=float, default=1.0, help="UV L1 regularizer weight (quality loss only)."
    )

    p.add_argument(
        "--w-vif-floor",
        type=float,
        default=50.0,
        help="Quadratic penalty weight for VIF below --vif-target.",
    )
    p.add_argument(
        "--vif-target", type=float, default=0.7, help="VIF floor target enforced by --w-vif-floor."
    )
    p.add_argument(
        "--w-unique-floor",
        type=float,
        default=10.0,
        help="Quadratic penalty weight for UNIQUE below --unique-target.",
    )
    p.add_argument(
        "--unique-target",
        type=float,
        default=0.0,
        help="UNIQUE floor target enforced by --w-unique-floor.",
    )
    p.add_argument(
        "--scene-aware-train",
        action="store_true",
        help="Use scene_id to apply per-scene non-diff ISP params "
        "during training, matching per-scene validation.",
    )
    p.add_argument(
        "--balance-scenes", action="store_true", help="Use a scene-balanced WeightedRandomSampler."
    )
    p.add_argument("--day-loss-weight", type=float, default=1.0)
    p.add_argument("--night-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--isp-reg-weight",
        type=float,
        default=0.0,
        help="Soft regularization weight that keeps trainable ISP "
        "params near their initial values. 0 disables it.",
    )
    p.add_argument("--isp-reg-gamma-scale", type=float, default=0.1)
    p.add_argument("--isp-reg-ccm-scale", type=float, default=0.05)

    p.add_argument(
        "--isp-reg-continuous-scale",
        type=float,
        default=0.1,
        help="Anti-drift scale for [0,1] knobs (LTM a/b/"
        "target_mean/detail_threshold, hist target_mean/"
        "std, sharp amount/threshold, raw_y blends, AWB "
        "lum_mask thresholds).",
    )
    p.add_argument(
        "--isp-reg-gain-scale",
        type=float,
        default=10.0,
        help="Anti-drift scale for gain-like knobs (ltm_detail_gain, awb_max_gain).",
    )
    p.add_argument(
        "--isp-reg-eps-log-scale",
        type=float,
        default=1.0,
        help="Anti-drift scale (in log-space) for guided-"
        "filter eps knobs (ltm_eps, post_denoise_eps, "
        "denoise_eps).",
    )

    p.add_argument(
        "--best-criterion",
        choices=["val_loss", "composite"],
        default="composite",
        help="val_loss = L1-style lower-is-better; "
        "composite = normalized metric score higher-is-better.",
    )
    p.add_argument(
        "--vif-guard-ratio",
        type=float,
        default=0.0,
        help="If >0, best-by-composite uses a VIF-guarded score: "
        "composite - penalty * sum(max(0, ratio*baseline_vif - scene_vif)).",
    )
    p.add_argument("--vif-guard-penalty", type=float, default=10.0)
    p.add_argument(
        "--norm-weights",
        default="artifacts/baselines/norm_weights.json",
        help="JSON file with frozen ranges/weights for the composite score. "
        "Required when --best-criterion=composite.",
    )
    p.add_argument(
        "--pretrain-eval-json",
        default=None,
        help="Optional path to pretrain_eval_metrics.json. When "
        "omitted, the script looks next to --ckpt and then in "
        "artifacts/checkpoints/cnn_pretrain/.",
    )
    p.add_argument("--eval-every", type=int, default=5, help="Full evaluation every N epochs")
    p.add_argument(
        "--eval-max-frames", type=int, default=3, help="Max frames per scene during eval"
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=10, help="Save periodic checkpoint every N epochs"
    )
    p.add_argument(
        "--max-train-batches", type=int, default=None, help="For dry run: limit batches per epoch"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="artifacts/checkpoints")
    p.add_argument("--history-name", default="e2e_history.json")
    p.add_argument("--ckpt-prefix", default="e2e")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from {ckpt_prefix}_resume.pth in output-dir if present",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


def compute_vif_guarded_score(
    composite: float, eval_results: dict, baseline: dict, ratio: float, penalty: float
) -> tuple[float, float]:
    """Penalize composite scores that are achieved by collapsing scene VIF."""
    if ratio <= 0.0:
        return float(composite), 0.0

    violation = 0.0
    for scene in ("day", "night"):
        result = eval_results.get(scene)
        baseline_scene = baseline.get(scene)
        if result is None or baseline_scene is None:
            continue
        floor = float(ratio) * float(baseline_scene["vif"])
        violation += max(0.0, floor - float(result["vif"]))

    guarded = float(composite) - float(penalty) * violation
    return guarded, violation


def make_full_scene_params(config: dict, device: str) -> dict[int, dict]:
    """Build full per-scene structural ISP parameter dicts."""
    default_isp = ISPPipeline(config, device=device)
    default_params = dict(default_isp.params)
    scene_name_to_id = {"day": 0, "night": 1, "tunnel": 2}
    full = {}
    for scene_name, scene_id in scene_name_to_id.items():
        params = {**default_params, **ISP_PARAMS.get(scene_name, {})}
        for key in TRAINABLE_ISP_PARAM_KEYS:
            params.pop(key, None)
        full[scene_id] = params
    return full


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unfreeze_isp(isp):
    """Set requires_grad=True on all ISP nn.Parameters."""
    n_tensors = 0
    n_scalars = 0
    for _, param in isp.named_parameters():
        param.requires_grad = True
        n_tensors += 1
        n_scalars += param.numel()
    return n_tensors, n_scalars


def snapshot_isp_params(isp) -> dict:
    """Snapshot every trainable ISP scalar / tensor as plain Python values."""
    return {
        "ccm_matrix": isp.ccm.ccm.detach().cpu().T.tolist(),
        "gamma": 1.0 / float(isp.gamma.inv_gamma.detach().cpu().item()),
        "ltm_a": float(isp.ltm.a.detach().cpu().item()),
        "ltm_b": float(isp.ltm.b.detach().cpu().item()),
        "ltm_eps": float(isp.ltm.eps.detach().cpu().item()),
        "ltm_target_mean": float(isp.ltm.target_mean.detach().cpu().item()),
        "ltm_detail_gain": float(isp.ltm.detail_gain.detach().cpu().item()),
        "ltm_detail_threshold": float(isp.ltm.detail_threshold.detach().cpu().item()),
        "awb_max_gain": float(isp.awb.max_gain.detach().cpu().item()),
        "awb_lum_mask_low": float(isp.awb.lum_mask_low.detach().cpu().item()),
        "awb_lum_mask_high": float(isp.awb.lum_mask_high.detach().cpu().item()),
        "hist_target_mean": float(isp.hist_norm.target_mean.detach().cpu().item()),
        "hist_target_std": float(isp.hist_norm.target_std.detach().cpu().item()),
        "post_denoise_eps": float(isp.post_denoise.eps.detach().cpu().item()),
        "sharp_amount": float(isp.sharpening.amount.detach().cpu().item()),
        "sharp_threshold": float(isp.sharpening.threshold.detach().cpu().item()),
        "denoise_eps": float(isp.denoise.eps.detach().cpu().item()),
        "raw_y_blend": float(isp.rgb2yuv.raw_y_blend.detach().cpu().item()),
        "raw_y_full_blend": float(isp.rgb2yuv.raw_y_full_blend.detach().cpu().item()),
    }


def snapshot_grad_means(isp) -> dict:
    """Snapshot grad means of the three trainable ISP params (post-backward)."""

    def _gm(p):
        if p is None or p.grad is None:
            return None
        return float(p.grad.abs().mean().item())

    return {
        "ccm_grad_mean": _gm(isp.ccm.ccm),
        "gamma_grad_mean": _gm(isp.gamma.inv_gamma),
    }


def isp_params_sane(params: dict) -> (bool, str):
    """Check the trainable params stay in sensible ranges."""
    reasons = []

    ccm = params["ccm_matrix"]
    flat = [v for row in ccm for v in row]
    if any(not math.isfinite(v) for v in flat):
        reasons.append("ccm has non-finite values")
    if any(v < -3.0 or v > 3.0 for v in flat):
        reasons.append(f"ccm has out-of-range element (min={min(flat):.3f}, max={max(flat):.3f})")

    g = params["gamma"]
    if not math.isfinite(g) or not (0.5 < g < 5.0):
        reasons.append(f"gamma={g:.3f} outside (0.5, 5.0)")

    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def build_scene_isp_with_learned(config, device, scene_name: str, learned: dict) -> ISPPipeline:
    """Build a per-scene ISP for evaluation."""
    scene_params = {
        key: value
        for key, value in ISP_PARAMS.get(scene_name, {}).items()
        if key not in TRAINABLE_ISP_PARAM_KEYS
    }
    isp = ISPPipeline(config, device=device, **scene_params)

    learned_overrides = {
        key: value
        for key, value in learned.items()
        if key in TRAINABLE_ISP_PARAM_KEYS and value is not None
    }
    if learned_overrides:
        learned_overrides = ISPPipeline.sanitize_trainable_params_dict(learned_overrides)
        isp.set_params(**learned_overrides)
    isp.eval()
    return isp


def run_per_scene_eval(
    cnn, isp_train, config, config_path, splits_json, split_name, device, max_frames
):
    """Full per-scene evaluation."""
    learned = snapshot_isp_params(isp_train)
    all_results = {}
    for scene_name in ISP_PARAMS.keys():
        all_items = load_split_items(str(splits_json), split_name)
        scene_items = [it for it in all_items if it["scene"] == scene_name]
        if not scene_items:
            continue
        scene_items = limit_eval_items(scene_items, max_frames)

        isp_scene = build_scene_isp_with_learned(
            config=config,
            device=device,
            scene_name=scene_name,
            learned=learned,
        )
        result = evaluate(
            isp=isp_scene,
            model=cnn,
            eval_items=scene_items,
            config_path=str(config_path),
            device=device,
            compute_iqa=True,
            verbose=False,
        )
        all_results[scene_name] = result

        del isp_scene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_results


def print_eval_results(results, baseline):
    print(
        f"\n  {'Scene':<8s} {'VIF':>8s} {'(d)':>8s} {'NRQM':>8s} {'(d)':>8s} "
        f"{'UNIQUE':>8s} {'(d)':>8s} {'L1_Y':>8s} {'L1_UV':>8s}"
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


def print_isp_params(params: dict, grad_means: dict):
    ccm = params["ccm_matrix"]
    print(f"  ISP params — gamma={params['gamma']:.4f}")
    print("  CCM:")
    for row in ccm:
        print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    if grad_means.get("ccm_grad_mean") is not None:
        print(
            f"  Grad means — ccm={grad_means['ccm_grad_mean']:.2e}  "
            f"gamma={grad_means['gamma_grad_mean']:.2e}"
        )


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 8)
    if isinstance(obj, (np.floating, np.integer)):
        return make_serializable(obj.item())
    return obj


def save_checkpoint(path: Path, epoch: int, isp, cnn, optimizer, val_loss, val_metrics):
    payload = {
        "epoch": epoch,
        "cnn_state_dict": cnn.state_dict(),
        "isp_state_dict": isp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "isp_params_trainable": snapshot_isp_params(isp),
    }
    _atomic_torch_save(payload, path)


def _atomic_torch_save(payload, path: Path):
    """Write to tmp file then rename — avoids half-written files on kill."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        torch.save(payload, path)
        try:
            tmp.unlink()
        except OSError:
            pass


def _atomic_json_save(obj, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        try:
            tmp.unlink()
        except OSError:
            pass


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    def _as_cpu_bytes(t):
        if isinstance(t, torch.Tensor):
            return t.detach().to("cpu").to(torch.uint8)
        return t

    torch.set_rng_state(_as_cpu_bytes(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_cpu_bytes(s) for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def save_resume_checkpoint(
    path: Path,
    epoch: int,
    isp,
    cnn,
    optimizer,
    best_val_loss,
    best_epoch,
    history,
    best_composite=float("-inf"),
    best_guarded_composite=float("-inf"),
):
    """Full-state checkpoint for resuming across Colab session breaks."""
    payload = {
        "epoch": epoch,
        "cnn_state_dict": cnn.state_dict(),
        "isp_state_dict": isp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_composite": best_composite,
        "best_guarded_composite": best_guarded_composite,
        "best_epoch": best_epoch,
        "history": history,
        "rng_state": _rng_state(),
    }
    _atomic_torch_save(payload, path)


def try_load_resume_checkpoint(path: Path, device: str):
    """Return the resume checkpoint dict if the file exists, else None."""
    if not Path(path).exists():
        return None
    return torch.load(str(path), map_location="cpu", weights_only=False)


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

    config_path = resolve(args.config)
    config = read_config(str(config_path), device=device)

    train_h5 = resolve(args.train_h5)
    splits_json = resolve(args.splits_json)
    ckpt_path = resolve(args.ckpt)
    try:
        baseline, baseline_path = load_e2e_minival_baseline(
            root=ROOT,
            ckpt_path=ckpt_path,
            path=args.pretrain_eval_json,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print(
            "E2E validation needs pretrain_eval_metrics.json so the VIF guard "
            "uses the same mini-val ISP baseline as the warm-start stage."
        )
        sys.exit(1)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"ERROR: pretrained CNN checkpoint not found at {ckpt_path}")
        print("Run scripts/run_pretrain_cnn.py first to produce it.")
        sys.exit(1)

    print("=" * 70)
    print("End-to-end ISP+CNN training")
    print("=" * 70)
    print(f"Device: {device}  Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}  Epochs: {args.epochs}")
    print(f"lr_isp: {args.lr_isp}  lr_cnn: {args.lr_cnn}  lambda_uv: {args.lambda_uv}")
    print(f"Loss: {args.loss}  |  Best criterion: {args.best_criterion}")
    print(f"Mini-val baseline source: {baseline_path}")

    norm_weights_path = resolve(args.norm_weights)
    try:
        composite_cfg = load_composite_config(norm_weights_path)
        print(
            "Composite: VIF_norm + "
            f"{composite_cfg['a']:.4f}*NRQM_norm + "
            f"{composite_cfg['b']:.4f}*UNIQUE_norm "
            f"({composite_cfg['mode']}) from {norm_weights_path.name}"
        )
    except FileNotFoundError:
        if args.best_criterion == "composite":
            print(f"ERROR: --best-criterion=composite requires {norm_weights_path}")
            sys.exit(1)
        composite_cfg = None
        print("[warn] norm_weights.json missing; composite score will not be logged.")

    if args.loss == "quality":
        quality_weights = QualityLossWeights(
            w_ssim=args.w_ssim,
            w_vif=args.w_vif,
            w_unique=args.w_unique,
            w_l1_y=args.w_l1_y,
            w_uv=args.w_uv,
            w_vif_floor=args.w_vif_floor,
            vif_target=args.vif_target,
            w_unique_floor=args.w_unique_floor,
            unique_target=args.unique_target,
        )
        print(
            f"Quality weights: w_ssim={quality_weights.w_ssim}  "
            f"w_vif={quality_weights.w_vif}  w_unique={quality_weights.w_unique}  "
            f"w_l1_y={quality_weights.w_l1_y}  "
            f"w_uv={quality_weights.w_uv}"
        )
        print(
            f"  floor penalties: "
            f"vif >= {quality_weights.vif_target} weight={quality_weights.w_vif_floor}, "
            f"unique >= {quality_weights.unique_target} weight={quality_weights.w_unique_floor}"
        )
    else:
        quality_weights = None

    print("\n[setup] Building training ISP with day params and unfreezing...")
    isp = ISPPipeline(config, device=device, **ISP_PARAMS["day"])
    n_tensors, n_scalars = unfreeze_isp(isp)
    print(
        f"  ISP unfrozen: {n_tensors} tensors, {n_scalars} scalar params "
        f"(ccm, gamma, and per-stage trainable knobs)"
    )

    print(f"\n[setup] Loading pretrained CNN from {ckpt_path}...")
    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn.train()
    print(f"  CNN loaded from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  CNN params: {count_trainable_parameters(cnn):,}")

    isp_params = [p for p in isp.parameters() if p.requires_grad]
    cnn_params = [p for p in cnn.parameters() if p.requires_grad]
    optimizer = Adam(
        [
            {"params": isp_params, "lr": args.lr_isp, "name": "isp"},
            {"params": cnn_params, "lr": args.lr_cnn, "name": "cnn"},
        ]
    )
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  optim group '{g['name']}': lr={g['lr']}, {len(g['params'])} tensors, {n} scalars")

    train_loader = create_dataloader(
        h5_path=str(train_h5),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        return_meta=args.scene_aware_train
        or args.balance_scenes
        or args.night_loss_weight != 1.0
        or args.day_loss_weight != 1.0,
        balance_scenes=args.balance_scenes,
    )
    n_total_batches = len(train_loader)
    print(f"\nTrain batches: {n_total_batches} ({len(train_loader.dataset)} patches)")
    scene_params_by_id = make_full_scene_params(config, device) if args.scene_aware_train else None
    scene_loss_weights_by_id = None
    if args.day_loss_weight != 1.0 or args.night_loss_weight != 1.0:
        scene_loss_weights_by_id = {
            0: float(args.day_loss_weight),
            1: float(args.night_loss_weight),
        }
    if args.scene_aware_train:
        print("Scene-aware training: ON (per-scene ISP non-diff knobs)")
    if args.balance_scenes:
        print("Scene-balanced sampler: ON")
    if scene_loss_weights_by_id is not None:
        print(f"Scene loss weights: day={args.day_loss_weight}  night={args.night_loss_weight}")

    best_val_loss = float("inf")
    best_composite = float("-inf")
    best_guarded_composite = float("-inf")
    best_epoch = -1
    history = []
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))
    resume_path = output_dir / f"{args.ckpt_prefix}_resume.pth"
    history_path = output_dir / args.history_name
    start_epoch = 1
    isp_anchor_params = snapshot_isp_params(isp)
    if args.isp_reg_weight > 0.0:
        print(f"ISP anchor regularization: weight={args.isp_reg_weight}")
        print(
            f"  per-tensor scales:  ccm={args.isp_reg_ccm_scale}  gamma={args.isp_reg_gamma_scale}"
        )
        print(
            f"  group scales:       continuous={args.isp_reg_continuous_scale}  "
            f"gain={args.isp_reg_gain_scale}  "
            f"eps_log={args.isp_reg_eps_log_scale}"
        )
    if args.vif_guard_ratio > 0.0:
        print(
            f"VIF-guarded composite selection: ratio={args.vif_guard_ratio}  "
            f"penalty={args.vif_guard_penalty}"
        )

    if args.resume:
        resumed = try_load_resume_checkpoint(resume_path, device=device)
        if resumed is None:
            print(f"\n[resume] No resume checkpoint at {resume_path}; starting fresh.")
        else:
            cnn.load_state_dict(resumed["cnn_state_dict"])
            isp.load_state_dict(resumed["isp_state_dict"])
            if hasattr(isp, "project_trainable_params_"):
                isp.project_trainable_params_()
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            best_val_loss = resumed.get("best_val_loss", float("inf"))
            best_composite = resumed.get("best_composite", float("-inf"))
            best_guarded_composite = resumed.get(
                "best_guarded_composite",
                float("-inf"),
            )
            best_epoch = resumed["best_epoch"]
            history = list(resumed["history"])
            if history and "isp_params" in history[0]:
                isp_anchor_params = history[0]["isp_params"]
            _restore_rng_state(resumed["rng_state"])
            start_epoch = int(resumed["epoch"]) + 1
            print(f"\n[resume] Loaded {resume_path}")
            print(
                f"[resume] Resuming at epoch {start_epoch} "
                f"(completed through epoch {resumed['epoch']})"
            )
            print(
                f"[resume] best_val_loss={best_val_loss:.6f} "
                f"best_composite={best_composite:.6f} "
                f"best_guarded={best_guarded_composite:.6f} "
                f"@ epoch {best_epoch}"
            )
            if start_epoch > args.epochs:
                print(f"[resume] Already completed {args.epochs} epochs. Nothing to do.")
                return

    print(f"\nStarting E2E training: epochs {start_epoch}..{args.epochs}")
    print(
        f"Eval every {args.eval_every} epochs  |  Checkpoint every {args.checkpoint_every} epochs\n"
    )

    if start_epoch == 1 and not history:
        init_params = snapshot_isp_params(isp)
        print_isp_params(init_params, grad_means={"ccm_grad_mean": None, "gamma_grad_mean": None})
        history.append({"epoch": 0, "isp_params": init_params})

    aborted = False
    for epoch in range(start_epoch, args.epochs + 1):
        cnn.train()
        isp.train()

        epoch_loss = 0.0
        epoch_l1y = 0.0
        epoch_l1uv = 0.0
        epoch_vif = 0.0
        epoch_ms_ssim = 0.0
        epoch_unique = 0.0
        epoch_isp_anchor_reg = 0.0
        n_ms_ssim = 0
        n_unique = 0
        n_isp_anchor_reg = 0
        last_grads = {}
        n_batches = 0

        t0 = time.time()
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}", leave=False, total=n_total_batches
        )

        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            metrics = e2e_train_step(
                isp=isp,
                cnn=cnn,
                optimizer=optimizer,
                batch=batch,
                pattern=pattern,
                lambda_uv=args.lambda_uv,
                loss_type=args.loss,
                quality_weights=quality_weights,
                isp_anchor_params=isp_anchor_params,
                isp_reg_weight=args.isp_reg_weight,
                isp_reg_gamma_scale=args.isp_reg_gamma_scale,
                isp_reg_ccm_scale=args.isp_reg_ccm_scale,
                isp_reg_continuous_scale=args.isp_reg_continuous_scale,
                isp_reg_gain_scale=args.isp_reg_gain_scale,
                isp_reg_eps_log_scale=args.isp_reg_eps_log_scale,
                scene_params_by_id=scene_params_by_id,
                scene_loss_weights_by_id=scene_loss_weights_by_id,
            )

            if not math.isfinite(metrics["loss"]):
                print(
                    f"\nABORT: non-finite loss ({metrics['loss']}) "
                    f"at epoch {epoch}, batch {batch_idx}"
                )
                aborted = True
                break

            epoch_loss += metrics["loss"]
            if metrics["l1_y"] is not None:
                epoch_l1y += metrics["l1_y"]
            if metrics["l1_uv"] is not None:
                epoch_l1uv += metrics["l1_uv"]
            if metrics["vif"] is not None:
                epoch_vif += metrics["vif"]
            if metrics.get("ms_ssim") is not None:
                epoch_ms_ssim += metrics["ms_ssim"]
                n_ms_ssim += 1
            if metrics.get("unique") is not None:
                epoch_unique += metrics["unique"]
                n_unique += 1
            if metrics.get("isp_anchor_reg") is not None:
                epoch_isp_anchor_reg += metrics["isp_anchor_reg"]
                n_isp_anchor_reg += 1
            last_grads = {
                "ccm_grad_mean": metrics["isp_ccm_grad_mean"],
                "gamma_grad_mean": metrics["isp_gamma_grad_mean"],
                "cnn_head_grad_mean": metrics["head_grad_mean"],
                "cnn_body_grad_mean": metrics["body_grad_mean"],
                "cnn_tail_grad_mean": metrics["tail_grad_mean"],
            }
            n_batches += 1

            if metrics.get("ms_ssim") is not None:
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.5f}",
                    ssim=f"{metrics['ms_ssim']:.4f}",
                    vif=f"{metrics['vif']:.4f}",
                )
            else:
                pbar.set_postfix(loss=f"{metrics['loss']:.5f}", l1y=f"{metrics['l1_y']:.5f}")

        pbar.close()
        elapsed = time.time() - t0

        if aborted:
            break
        if n_batches == 0:
            print(f"Epoch {epoch}: no batches processed, aborting.")
            break

        avg_loss = epoch_loss / n_batches
        avg_l1y = epoch_l1y / n_batches
        avg_l1uv = epoch_l1uv / n_batches
        avg_vif = epoch_vif / n_batches
        avg_ms_ssim = (epoch_ms_ssim / n_ms_ssim) if n_ms_ssim > 0 else None
        avg_unique = (epoch_unique / n_unique) if n_unique > 0 else None
        avg_isp_anchor_reg = (
            epoch_isp_anchor_reg / n_isp_anchor_reg if n_isp_anchor_reg > 0 else None
        )

        current_params = snapshot_isp_params(isp)
        sane, reason = isp_params_sane(current_params)

        if avg_ms_ssim is not None:
            print(
                f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.6f}  "
                f"ms_ssim={avg_ms_ssim:.4f}  vif={avg_vif:.4f}  "
                f"unique={avg_unique:.4f}  l1_uv={avg_l1uv:.6f}  "
                f"isp_reg={avg_isp_anchor_reg if avg_isp_anchor_reg is not None else 0.0:.4f}  "
                f"({elapsed:.1f}s)"
            )
        else:
            print(
                f"Epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.6f}  "
                f"l1_y={avg_l1y:.6f}  l1_uv={avg_l1uv:.6f}  "
                f"train_vif={avg_vif:.4f}  ({elapsed:.1f}s)"
            )
        print_isp_params(current_params, last_grads)
        if not sane:
            print(f"  WARNING: ISP params out of sane range: {reason}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_l1y": avg_l1y,
            "train_l1uv": avg_l1uv,
            "train_vif_monitor": avg_vif,
            "train_ms_ssim": avg_ms_ssim,
            "train_unique": avg_unique,
            "train_isp_anchor_reg": avg_isp_anchor_reg,
            "elapsed_s": elapsed,
            "isp_params": current_params,
            "grad_means": last_grads,
            "isp_params_sane": bool(sane),
            "isp_sane_reason": reason,
        }

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            print(f"\n  Running full evaluation (epoch {epoch})...")
            cnn.eval()
            eval_results = run_per_scene_eval(
                cnn=cnn,
                isp_train=isp,
                config=config,
                config_path=config_path,
                splits_json=splits_json,
                split_name="val",
                device=device,
                max_frames=args.eval_max_frames,
            )
            print_eval_results(eval_results, baseline)

            val_scenes = [r for r in eval_results.values() if r is not None]
            if val_scenes:
                val_l1y = sum(r["l1_y"] for r in val_scenes) / len(val_scenes)
                val_l1uv = sum(r["l1_uv"] for r in val_scenes) / len(val_scenes)
                val_vif = sum(r["vif"] for r in val_scenes) / len(val_scenes)
                val_nrqm = sum(r["nrqm"] for r in val_scenes) / len(val_scenes)
                val_unique = sum(r["unique"] for r in val_scenes) / len(val_scenes)
                val_loss = val_l1y + args.lambda_uv * val_l1uv

                if composite_cfg is not None:
                    val_composite_terms = compute_composite_terms(
                        val_vif,
                        val_nrqm,
                        val_unique,
                        composite_cfg,
                    )
                    val_composite = compute_composite(
                        val_vif,
                        val_nrqm,
                        val_unique,
                        composite_cfg,
                    )
                    val_guarded_composite, val_vif_guard_violation = compute_vif_guarded_score(
                        composite=val_composite,
                        eval_results=eval_results,
                        baseline=baseline,
                        ratio=args.vif_guard_ratio,
                        penalty=args.vif_guard_penalty,
                    )
                else:
                    val_composite_terms = None
                    val_composite = None
                    val_guarded_composite = None
                    val_vif_guard_violation = None

                epoch_record.update(
                    {
                        "val_loss": val_loss,
                        "val_l1y": val_l1y,
                        "val_l1uv": val_l1uv,
                        "val_vif": val_vif,
                        "val_nrqm": val_nrqm,
                        "val_unique": val_unique,
                        "val_vif_term": (
                            val_composite_terms["vif_term"] if val_composite_terms else None
                        ),
                        "val_a_nrqm_term": (
                            val_composite_terms["a_nrqm_term"] if val_composite_terms else None
                        ),
                        "val_b_unique_term": (
                            val_composite_terms["b_unique_term"] if val_composite_terms else None
                        ),
                        "val_composite": val_composite,
                        "val_guarded_composite": val_guarded_composite,
                        "val_vif_guard_violation": val_vif_guard_violation,
                        "val_per_scene": eval_results,
                    }
                )

                val_loss_is_best = val_loss < best_val_loss
                composite_is_best = val_composite is not None and val_composite > best_composite
                guarded_is_best = (
                    val_guarded_composite is not None
                    and val_guarded_composite > best_guarded_composite
                )

                if val_loss_is_best:
                    best_val_loss = val_loss
                if composite_is_best:
                    best_composite = val_composite
                if guarded_is_best:
                    best_guarded_composite = val_guarded_composite

                if args.best_criterion == "composite":
                    if val_composite is None:
                        print(
                            "\n  [warn] composite unavailable; falling back "
                            "to val_loss for best selection this epoch."
                        )
                        is_best = val_loss <= best_val_loss
                        chosen_value = val_loss
                        chosen_name = "val_loss"
                    else:
                        if args.vif_guard_ratio > 0.0:
                            is_best = guarded_is_best
                            chosen_value = val_guarded_composite
                            chosen_name = "guarded_composite"
                        else:
                            is_best = composite_is_best
                            chosen_value = val_composite
                            chosen_name = "composite"
                else:
                    is_best = val_loss_is_best
                    chosen_value = val_loss
                    chosen_name = "val_loss"

                if is_best:
                    best_epoch = epoch
                    best_path = output_dir / f"{args.ckpt_prefix}_best.pth"
                    save_checkpoint(
                        best_path,
                        epoch,
                        isp,
                        cnn,
                        optimizer,
                        val_loss=val_loss,
                        val_metrics={
                            "l1_y": val_l1y,
                            "l1_uv": val_l1uv,
                            "vif": val_vif,
                            "nrqm": val_nrqm,
                            "unique": val_unique,
                            "vif_term": (
                                val_composite_terms["vif_term"] if val_composite_terms else None
                            ),
                            "a_nrqm_term": (
                                val_composite_terms["a_nrqm_term"] if val_composite_terms else None
                            ),
                            "b_unique_term": (
                                val_composite_terms["b_unique_term"]
                                if val_composite_terms
                                else None
                            ),
                            "composite": val_composite,
                            "guarded_composite": val_guarded_composite,
                            "vif_guard_violation": val_vif_guard_violation,
                            "best_criterion": chosen_name,
                            "best_criterion_value": chosen_value,
                        },
                    )
                    if val_composite is not None:
                        print(
                            f"\n  *** New best {chosen_name}={chosen_value:.6f} "
                            f"(VIF={val_vif:.4f}, NRQM={val_nrqm:.4f}, "
                            f"UNIQUE={val_unique:.4f}, "
                            f"b*UNIQUE={val_composite_terms['b_unique_term']:.4f}, "
                            f"composite={val_composite:.4f}, "
                            f"guarded={val_guarded_composite:.4f}) "
                            f"— saved to {best_path}"
                        )
                    else:
                        print(
                            f"\n  *** New best {chosen_name}={chosen_value:.6f} "
                            f"(VIF={val_vif:.4f}) — saved to {best_path}"
                        )
                else:
                    if args.best_criterion == "composite" and val_composite is not None:
                        print(
                            f"\n  composite={val_composite:.4f} "
                            f"guarded={val_guarded_composite:.4f} "
                            f"(best={best_composite:.4f}, "
                            f"guarded_best={best_guarded_composite:.4f} "
                            f"@ epoch {best_epoch})  "
                            f"val_loss={val_loss:.6f}"
                        )
                    else:
                        print(
                            f"\n  val_loss={val_loss:.6f} "
                            f"(best={best_val_loss:.6f} @ epoch {best_epoch})"
                        )

        if epoch % args.checkpoint_every == 0:
            pc_path = output_dir / f"{args.ckpt_prefix}_epoch_{epoch:02d}.pth"
            save_checkpoint(
                pc_path,
                epoch,
                isp,
                cnn,
                optimizer,
                val_loss=epoch_record.get("val_loss"),
                val_metrics={
                    k: epoch_record.get(k)
                    for k in (
                        "val_l1y",
                        "val_l1uv",
                        "val_vif",
                        "val_nrqm",
                        "val_unique",
                        "val_vif_term",
                        "val_a_nrqm_term",
                        "val_b_unique_term",
                        "val_composite",
                        "val_guarded_composite",
                    )
                }
                if epoch_record.get("val_loss") is not None
                else None,
            )
            print(f"  Periodic checkpoint: {pc_path}")

        history.append(epoch_record)

        save_resume_checkpoint(
            resume_path,
            epoch=epoch,
            isp=isp,
            cnn=cnn,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            best_composite=best_composite,
            best_guarded_composite=best_guarded_composite,
            best_epoch=best_epoch,
            history=history,
        )
        _atomic_json_save(make_serializable(history), history_path)

        if not sane:
            print(f"\nABORT: ISP params left sane range at epoch {epoch}: {reason}")
            aborted = True
            break

    print(f"\n{'=' * 60}")
    if aborted:
        print("TRAINING ABORTED")
    else:
        print("E2E TRAINING (1ST HALF) COMPLETE")
    print(f"{'=' * 60}")
    if best_epoch > 0:
        if args.best_criterion == "composite" and math.isfinite(best_composite):
            print(
                f"Best epoch: {best_epoch}  composite={best_composite:.4f}  "
                f"guarded={best_guarded_composite:.4f}  "
                f"val_loss={best_val_loss:.6f}"
            )
        else:
            print(
                f"Best epoch: {best_epoch}  val_loss={best_val_loss:.6f}  "
                f"composite={best_composite:.4f}"
            )
        print(f"Best criterion: {args.best_criterion}")
        print(f"Checkpoint: {output_dir / (args.ckpt_prefix + '_best.pth')}")
    else:
        print("No validation ran (0 val epochs reached). No best checkpoint.")

    _atomic_json_save(make_serializable(history), history_path)
    print(f"History: {history_path}")
    print(f"Resume checkpoint: {resume_path}")

    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
