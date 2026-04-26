"""
Per-term gradient inspector for the quality-aligned loss.
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from isp.color.conversions import yuv420_to_rgb_bt709_full
from isp.config.config_reader import read_config
from isp.data.dataset_utils import create_dataloader
from isp.models.residual_cnn import ResidualCNN
from isp.pipeline.pipeline import ISPPipeline
from isp.training.quality_loss import (
    _get_pyiqa_metric,
    compute_vif_from_raw_and_y_diff,
)
from isp.training.training_utils import forward_isp_cnn_diff

ISP_PARAMS_DAY = dict(
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
)


def _grad_mean(p) -> float:
    if p is None or p.grad is None:
        return 0.0
    g = p.grad
    if not torch.isfinite(g).all():
        return float("nan")
    return float(g.abs().mean().item())


def _zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _fmt(x: float) -> str:
    if x != x:
        return "    NaN "
    if x == 0:
        return "  0      "
    return f"{x:8.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-h5", default="dataset/train_patches.h5")
    ap.add_argument("--config", default="data/imx623.toml")
    ap.add_argument("--ckpt", default="artifacts/checkpoints/cnn_pretrained.pth")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    # Defaults match run_e2e_train.py --loss quality.
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--w-vif", type=float, default=0.3)
    ap.add_argument("--w-unique", type=float, default=0.1)
    ap.add_argument("--w-l1-y", type=float, default=0.0)
    ap.add_argument("--w-uv", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[info] CUDA not available, using CPU.")

    config_path = ROOT / args.config
    config = read_config(str(config_path), device=device)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    print("=" * 76)
    print("Quality-loss per-term gradient inspector")
    print("=" * 76)
    print(f"Device: {device}  batch_size: {args.batch_size}  pattern: {pattern}")
    print(f"Config: {config_path.name}")
    print(
        f"Weights (current): w_ssim={args.w_ssim}  w_vif={args.w_vif}  "
        f"w_unique={args.w_unique}  w_l1_y={args.w_l1_y}  w_uv={args.w_uv}"
    )

    print("\n[setup] Building ISP (day params) with unfrozen CCM/gamma/saturation...")
    isp = ISPPipeline(config, device=device, **ISP_PARAMS_DAY)
    n_scalars = 0
    for _, p in isp.named_parameters():
        p.requires_grad = True
        n_scalars += p.numel()
    print(f"  Unfrozen {n_scalars} scalar params.")

    ckpt_path = ROOT / args.ckpt
    print(f"[setup] Loading pretrained CNN from {ckpt_path}...")
    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    cnn.load_state_dict(sd["model_state_dict"])
    cnn.train()
    isp.train()

    print(f"[setup] Loading one batch from {args.train_h5}...")
    loader = create_dataloader(
        h5_path=str(ROOT / args.train_h5),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        return_meta=False,
    )
    batch = next(iter(loader))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    print(
        f"  raw: {tuple(batch['raw'].shape)}  "
        f"y_ref: {tuple(batch['y_ref'].shape)}  "
        f"uv_ref: {tuple(batch['uv_ref'].shape)}"
    )

    tracked = {
        "isp_ccm": isp.ccm.ccm if hasattr(isp, "ccm") else None,
        "isp_gamma": isp.gamma.inv_gamma if hasattr(isp, "gamma") else None,
        "isp_saturation": (
            isp.saturation_adjust.saturation if hasattr(isp, "saturation_adjust") else None
        ),
        "cnn_head": cnn.head[0].weight if hasattr(cnn, "head") else None,
        "cnn_body0": (
            cnn.body[0].conv1.weight if hasattr(cnn, "body") and len(cnn.body) > 0 else None
        ),
        "cnn_tail": cnn.tail.weight if hasattr(cnn, "tail") else None,
    }
    tracked = {k: v for k, v in tracked.items() if v is not None}
    all_params = list(tracked.values())

    print("\n[forward] Running forward_isp_cnn_diff (ISP runs per patch)...")
    _zero_grads(all_params)
    outputs = forward_isp_cnn_diff(isp, cnn, batch["raw"])
    raw_12bit = outputs["raw_12bit"]
    y_pred = outputs["y_pred"]
    uv_pred = outputs["uv_pred"]
    y_ref = batch["y_ref"]
    uv_ref = batch["uv_ref"]
    print(f"  y_pred: {tuple(y_pred.shape)}  uv_pred: {tuple(uv_pred.shape)}")

    print("[terms] Computing MS_SSIM, VIF, UNIQUE, L1_UV, L1_Y...")
    ms_ssim_metric = _get_pyiqa_metric("ms_ssim", torch.device(device))
    unique_metric = _get_pyiqa_metric("unique", torch.device(device))

    ms_ssim_val = ms_ssim_metric(y_pred, y_ref).to(y_pred.dtype).mean()

    rgb_pred = yuv420_to_rgb_bt709_full(y_pred, uv_pred)
    unique_val = unique_metric(rgb_pred).to(y_pred.dtype).mean()

    vif_val = (
        compute_vif_from_raw_and_y_diff(
            raw_12bit,
            y_pred,
            pattern,
        )
        .to(y_pred.dtype)
        .mean()
    )

    l1_uv = F.l1_loss(uv_pred, uv_ref)
    l1_y = F.l1_loss(y_pred, y_ref)

    terms = [
        ("MS_SSIM", ms_ssim_val, args.w_ssim, True),
        ("VIF", vif_val, args.w_vif, True),
        ("UNIQUE", unique_val, args.w_unique, True),
        ("L1_Y", l1_y, args.w_l1_y, args.w_l1_y > 0.0),
        ("L1_UV", l1_uv, args.w_uv, True),
    ]

    print()
    print("Raw term values (pre-weighting):")
    for name, val, _w, _in in terms:
        tag = "  [quality loss]" if _in else "  [reference / v1 proxy]"
        print(f"  {name:<8s}  value = {val.item():+.6f}{tag}")

    print()
    print("[backward] Backward per term (retain_graph reused across terms)...")
    rows = []
    for i, (name, val, w, _in) in enumerate(terms):
        _zero_grads(all_params)
        keep = i < len(terms) - 1
        val.backward(retain_graph=keep)
        row = {"term": name, "weight": w, "in_quality": _in}
        for pname, p in tracked.items():
            row[pname] = _grad_mean(p)
        rows.append(row)

    pnames = list(tracked.keys())
    col_w = 11

    print()
    print("Unweighted per-parameter |grad|.mean() per term")
    print("-" * (13 + (col_w + 1) * len(pnames)))
    print(f"  {'term':<10s} " + " ".join(f"{n:>{col_w}s}" for n in pnames))
    for row in rows:
        print(f"  {row['term']:<10s} " + " ".join(f"{_fmt(row[n]):>{col_w}s}" for n in pnames))

    print()
    print("Weighted per-parameter |grad|.mean() per term (|w| * unweighted)")
    print("(L1_Y uses |w|=1 for reference; not in quality loss)")
    print("-" * (13 + (col_w + 1) * len(pnames)))
    print(f"  {'term':<10s} " + " ".join(f"{n:>{col_w}s}" for n in pnames))
    for row in rows:
        w_abs = abs(row["weight"]) if row["weight"] is not None else 1.0
        print(
            f"  {row['term']:<10s} " + " ".join(f"{_fmt(row[n] * w_abs):>{col_w}s}" for n in pnames)
        )

    print()
    print("Dominance ratio per parameter (max / min over quality terms, weighted):")
    quality_rows = [r for r in rows if r["in_quality"]]
    for n in pnames:
        vals = []
        for r in quality_rows:
            w_abs = abs(r["weight"]) if r["weight"] is not None else 1.0
            v = r[n] * w_abs
            if v > 0 and v == v:
                vals.append(v)
        if len(vals) >= 2:
            ratio = max(vals) / min(vals)
            flag = "  <-- imbalanced" if ratio > 10 else ""
            print(f"  {n:<16s}  ratio = {ratio:7.1f}x{flag}")

    print()
    print(
        "Rescale suggestion (balance quality terms so CCM weighted |grad|"
        " equals their geometric mean):"
    )
    log_grads = []
    for r in quality_rows:
        w_abs = abs(r["weight"]) if r["weight"] is not None else 1.0
        v = r["isp_ccm"] * w_abs
        if v > 0 and v == v:
            log_grads.append(math.log(v))
    if len(log_grads) >= 2:
        gm = math.exp(sum(log_grads) / len(log_grads))
        print(f"  Target CCM weighted |grad| (geometric mean) = {gm:.2e}")
        for r in quality_rows:
            w_abs = abs(r["weight"]) if r["weight"] is not None else 1.0
            v = r["isp_ccm"] * w_abs
            if v > 0 and v == v:
                new_w = w_abs * (gm / v)
                print(
                    f"  {r['term']:<8s}  current |w| = {w_abs:7.3f}  "
                    f"weighted CCM grad = {v:.2e}  -> suggested |w| = {new_w:7.3f}"
                )
            else:
                print(
                    f"  {r['term']:<8s}  current |w| = {w_abs:7.3f}  "
                    f"weighted CCM grad = zero/NaN (excluded from rescale)"
                )
    else:
        print("  Not enough finite non-zero CCM grads to rescale.")

    print()
    l1y = next((r for r in rows if r["term"] == "L1_Y"), None)
    l1uv = next((r for r in rows if r["term"] == "L1_UV"), None)
    if l1y and l1uv:
        print("Reference (v1 proxy loss was L1_y + lambda_uv * L1_uv):")
        print(f"  Unweighted CCM grad from L1_Y   = {_fmt(l1y['isp_ccm'])}")
        print(f"  Unweighted CCM grad from L1_UV  = {_fmt(l1uv['isp_ccm'])}")
        print("  If quality-loss weighted CCM grads sit in the same order of")
        print("  magnitude as these, v1 learning rates should transfer.")


if __name__ == "__main__":
    main()
