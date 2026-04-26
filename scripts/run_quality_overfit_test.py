"""
Overfit one batch with the quality-aligned loss.
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.optim import Adam

from isp.color.conversions import (
    yuv420_to_rgb_bt709_full,
    yuv420_to_yuv444,
    yuv444_to_yuv420,
)
from isp.config.config_reader import read_config
from isp.evaluation.composite_score import (
    compute_composite,
    compute_normalized_terms,
    load_composite_config,
)
from isp.models.residual_cnn import ResidualCNN, count_trainable_parameters
from isp.pipeline.pipeline import ISPPipeline
from isp.training.quality_loss import (
    QualityLossWeights,
    _get_pyiqa_metric,
    compute_quality_loss,
)
from isp.training.training_utils import forward_isp_cnn


def parse_args():
    parser = argparse.ArgumentParser(description="Quality-loss overfit test on one batch")
    parser.add_argument("--h5-path", type=str, default="dataset/train_patches.h5")
    parser.add_argument("--config", type=str, default="data/imx623.toml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr-values", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--w-ssim", type=float, default=1.0)
    parser.add_argument("--w-vif", type=float, default=0.3)
    parser.add_argument("--w-unique", type=float, default=0.3)
    parser.add_argument("--w-uv", type=float, default=0.1)
    parser.add_argument("--norm-weights", type=str, default="artifacts/baselines/norm_weights.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/overfit_test")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_batch_from_h5(h5_path: str, batch_size: int) -> dict:
    from isp.data.dataset_utils import ISPDataset

    dataset = ISPDataset(h5_path, return_meta=True)
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    dataset.close()

    return {
        "raw": torch.stack([sample["raw"] for sample in samples]),
        "y_ref": torch.stack([sample["y_ref"] for sample in samples]),
        "uv_ref": torch.stack([sample["uv_ref"] for sample in samples]),
    }


def load_batch_from_video(config: dict, batch_size: int, patch_size: int = 256) -> dict:
    """Fallback: read one frame from day_0 and cut deterministic patches."""
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader

    width = int(config["img"]["width"])
    height = int(config["img"]["height"])
    top, bottom = config["img"]["emb_lines"]
    out_h = height - top - bottom

    raw_path = str(ROOT / "data" / "day_0.bin")
    yuv_path = str(ROOT / "data" / "day_0.yuv")

    with (
        RAWVideoReader(raw_path, config, device="cpu") as raw_reader,
        NV12VideoReader(yuv_path, width, out_h, device="cpu") as yuv_reader,
    ):
        (raw_frame, _), (yuv_frame, _) = next(zip(raw_reader, yuv_reader, strict=True))

    y_plane, u_plane, v_plane = yuv_frame
    raw_np = raw_frame.cpu().numpy()
    y_np = y_plane[0, 0].cpu().numpy()
    u_np = u_plane[0, 0].cpu().numpy()
    v_np = v_plane[0, 0].cpu().numpy()

    ps = patch_size
    uv_ps = ps // 2
    coords = []
    for y in range(0, out_h - ps + 1, ps):
        for x in range(0, width - ps + 1, ps):
            if x % 2 == 0 and y % 2 == 0:
                coords.append((x, y))
    coords = coords[:batch_size]

    raws, y_refs, uv_refs = [], [], []
    for x, y in coords:
        raws.append(
            torch.from_numpy(raw_np[y : y + ps, x : x + ps].copy()).float().unsqueeze(0) / 4095.0
        )
        y_refs.append(
            torch.from_numpy(y_np[y : y + ps, x : x + ps].copy()).float().unsqueeze(0) / 255.0
        )
        ux, uy = x // 2, y // 2
        u_patch = u_np[uy : uy + uv_ps, ux : ux + uv_ps].copy()
        v_patch = v_np[uy : uy + uv_ps, ux : ux + uv_ps].copy()
        uv_refs.append(torch.from_numpy(np.stack([u_patch, v_patch], axis=0)).float() / 255.0)

    return {
        "raw": torch.stack(raws),
        "y_ref": torch.stack(y_refs),
        "uv_ref": torch.stack(uv_refs),
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def forward_cnn_from_isp(cnn, y_isp: torch.Tensor, uv_isp: torch.Tensor) -> dict:
    yuv444_isp = yuv420_to_yuv444(y_isp, uv_isp)
    residual = cnn(yuv444_isp)
    yuv444_pred = torch.clamp(yuv444_isp + residual, 0.0, 1.0)
    y_pred, uv_pred = yuv444_to_yuv420(yuv444_pred)
    return {
        "yuv444_isp": yuv444_isp,
        "residual": residual,
        "yuv444_pred": yuv444_pred,
        "y_pred": y_pred,
        "uv_pred": uv_pred,
    }


def grad_mean(parameter) -> float | None:
    if parameter is None or parameter.grad is None:
        return None
    return float(parameter.grad.abs().mean().item())


def metric_row(
    step: int,
    lr: float,
    cnn,
    raw_batch: torch.Tensor,
    y_isp: torch.Tensor,
    uv_isp: torch.Tensor,
    y_ref: torch.Tensor,
    uv_ref: torch.Tensor,
    pattern: str,
    weights: QualityLossWeights,
    composite_cfg: dict,
    device: torch.device,
    grads: dict | None = None,
) -> tuple[dict, dict]:
    with torch.no_grad():
        outputs = forward_cnn_from_isp(cnn, y_isp, uv_isp)
        loss_dict = compute_quality_loss(
            raw_batch=raw_batch,
            y_pred=outputs["y_pred"],
            uv_pred=outputs["uv_pred"],
            y_ref=y_ref,
            uv_ref=uv_ref,
            pattern=pattern,
            weights=weights,
            lambda_uv=None,
        )
        rgb_pred = yuv420_to_rgb_bt709_full(outputs["y_pred"], outputs["uv_pred"])
        nrqm_metric = _get_pyiqa_metric("nrqm", device)
        nrqm = float(nrqm_metric(rgb_pred).to(outputs["y_pred"].dtype).mean().item())

    vif = float(loss_dict["vif"].item())
    unique = float(loss_dict["unique"].item())
    norm_terms = compute_normalized_terms(vif, nrqm, unique, composite_cfg)
    composite = compute_composite(vif, nrqm, unique, composite_cfg)

    row = {
        "step": step,
        "lr": lr,
        "loss": float(loss_dict["loss"].item()),
        "ms_ssim": float(loss_dict["ms_ssim"].item()),
        "vif": vif,
        "unique": unique,
        "vif_norm": norm_terms["vif_norm"],
        "nrqm_norm": norm_terms["nrqm_norm"],
        "unique_norm": norm_terms["unique_norm"],
        "nrqm": nrqm,
        "composite": float(composite),
        "l1_y": float(loss_dict["l1_y"].item()),
        "l1_uv": float(loss_dict["l1_uv"].item()),
    }
    if grads:
        row.update(grads)
    return row, outputs


def train_one_lr(
    lr: float,
    initial_state: dict,
    batch: dict,
    y_isp: torch.Tensor,
    uv_isp: torch.Tensor,
    pattern: str,
    weights: QualityLossWeights,
    composite_cfg: dict,
    steps: int,
    device: torch.device,
) -> tuple[list[dict], dict, dict]:
    cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    cnn.load_state_dict(initial_state)
    optimizer = Adam(cnn.parameters(), lr=lr)

    raw = batch["raw"]
    y_ref = batch["y_ref"]
    uv_ref = batch["uv_ref"]

    history = []
    start_row, outputs_before = metric_row(
        step=0,
        lr=lr,
        cnn=cnn,
        raw_batch=raw,
        y_isp=y_isp,
        uv_isp=uv_isp,
        y_ref=y_ref,
        uv_ref=uv_ref,
        pattern=pattern,
        weights=weights,
        composite_cfg=composite_cfg,
        device=device,
    )
    history.append(start_row)

    for step in range(1, steps + 1):
        cnn.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = forward_cnn_from_isp(cnn, y_isp, uv_isp)
        loss_dict = compute_quality_loss(
            raw_batch=raw,
            y_pred=outputs["y_pred"],
            uv_pred=outputs["uv_pred"],
            y_ref=y_ref,
            uv_ref=uv_ref,
            pattern=pattern,
            weights=weights,
            lambda_uv=None,
        )
        loss = loss_dict["loss"]
        loss.backward()

        grads = {
            "head_grad_mean": grad_mean(cnn.head[0].weight),
            "body_grad_mean": grad_mean(cnn.body[0].conv1.weight),
            "tail_grad_mean": grad_mean(cnn.tail.weight),
        }

        optimizer.step()

        row, _ = metric_row(
            step=step,
            lr=lr,
            cnn=cnn,
            raw_batch=raw,
            y_isp=y_isp,
            uv_isp=uv_isp,
            y_ref=y_ref,
            uv_ref=uv_ref,
            pattern=pattern,
            weights=weights,
            composite_cfg=composite_cfg,
            device=device,
            grads=grads,
        )
        history.append(row)

        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"  step {step:3d}/{steps}: "
                f"loss={row['loss']:.6f}  "
                f"ssim={row['ms_ssim']:.4f}  "
                f"vif={row['vif']:.4f}  "
                f"unique={row['unique']:.4f}  "
                f"l1_uv={row['l1_uv']:.5f}  "
                f"composite={row['composite']:.4f}"
            )

    _, outputs_after = metric_row(
        step=steps,
        lr=lr,
        cnn=cnn,
        raw_batch=raw,
        y_isp=y_isp,
        uv_isp=uv_isp,
        y_ref=y_ref,
        uv_ref=uv_ref,
        pattern=pattern,
        weights=weights,
        composite_cfg=composite_cfg,
        device=device,
    )
    return history, outputs_before, outputs_after


def save_visualization(output_dir: Path, batch: dict, outputs_before: dict, outputs_after: dict):
    from PIL import Image

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    def to_u8(t):
        return (t.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    idx = 0
    Image.fromarray(to_u8(batch["y_ref"][idx, 0]), "L").save(viz_dir / "y_ref.png")
    Image.fromarray(to_u8(outputs_before["y_pred"][idx, 0]), "L").save(viz_dir / "y_before.png")
    Image.fromarray(to_u8(outputs_after["y_pred"][idx, 0]), "L").save(viz_dir / "y_after.png")

    Image.fromarray(to_u8(batch["uv_ref"][idx, 0]), "L").save(viz_dir / "u_ref.png")
    Image.fromarray(to_u8(outputs_before["uv_pred"][idx, 0]), "L").save(viz_dir / "u_before.png")
    Image.fromarray(to_u8(outputs_after["uv_pred"][idx, 0]), "L").save(viz_dir / "u_after.png")

    print(f"Visualizations saved to {viz_dir}")


def save_history_csv(path: Path, rows: list[dict]):
    columns = [
        "step",
        "lr",
        "loss",
        "ms_ssim",
        "vif",
        "vif_norm",
        "unique",
        "unique_norm",
        "nrqm",
        "nrqm_norm",
        "composite",
        "l1_y",
        "l1_uv",
        "head_grad_mean",
        "body_grad_mean",
        "tail_grad_mean",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col)
                if value is None:
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.8f}")
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def summarize_run(history: list[dict]) -> dict:
    start = history[0]
    final = history[-1]
    return {
        "lr": final["lr"],
        "loss_decreased": final["loss"] < start["loss"],
        "ms_ssim_increased": final["ms_ssim"] > start["ms_ssim"],
        "vif_increased": final["vif"] > start["vif"],
        "unique_increased": final["unique"] > start["unique"],
        "l1_uv_not_increased": final["l1_uv"] <= start["l1_uv"],
        "composite_not_decreased": final["composite"] >= start["composite"],
        "cnn_grads_present": all(
            final.get(name) is not None and final.get(name) > 0.0
            for name in ["head_grad_mean", "body_grad_mean", "tail_grad_mean"]
        ),
        "start": start,
        "final": final,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    config_path = resolve(args.config)
    config = read_config(str(config_path), device=device_name)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))
    composite_cfg = load_composite_config(resolve(args.norm_weights))

    h5_path = resolve(args.h5_path)
    if h5_path.exists():
        print(f"Loading batch from HDF5: {h5_path}")
        batch = load_batch_from_h5(str(h5_path), args.batch_size)
    else:
        print(f"HDF5 not found ({h5_path}), reading patches from video files")
        batch = load_batch_from_video(config, args.batch_size)

    batch = move_batch_to_device(batch, device)
    print(
        f"Batch shapes: raw={tuple(batch['raw'].shape)}, "
        f"y_ref={tuple(batch['y_ref'].shape)}, "
        f"uv_ref={tuple(batch['uv_ref'].shape)}"
    )

    weights = QualityLossWeights(
        w_ssim=args.w_ssim,
        w_vif=args.w_vif,
        w_unique=args.w_unique,
        w_uv=args.w_uv,
    )
    print(f"Quality weights: {weights}")
    print(
        "Composite: VIF_norm + "
        f"{composite_cfg['a']:.4f}*NRQM_norm + "
        f"{composite_cfg['b']:.4f}*UNIQUE_norm "
        f"({composite_cfg['mode']})"
    )

    isp = ISPPipeline(
        config,
        device=device_name,
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
    freeze_module(isp)

    print(f"CNN params: {count_trainable_parameters(ResidualCNN()):,}")
    print("Precomputing fixed ISP outputs...")
    with torch.no_grad():
        seed_cnn = ResidualCNN().to(device)
        isp_outputs = forward_isp_cnn(isp, seed_cnn, batch["raw"])
        y_isp = isp_outputs["y_isp"].detach()
        uv_isp = isp_outputs["uv_isp"].detach()
    del seed_cnn

    torch.manual_seed(args.seed)
    initial_cnn = ResidualCNN(
        in_channels=3, hidden_channels=32, out_channels=3, num_blocks=5, num_groups=8
    ).to(device)
    initial_state = deepcopy(initial_cnn.state_dict())
    del initial_cnn

    all_rows = []
    summaries = []
    best_final = None
    best_outputs = None

    for lr in args.lr_values:
        print(f"\n{'=' * 72}")
        print(f"Quality overfit test: lr={lr:g}, steps={args.steps}")
        print(f"{'=' * 72}")
        t0 = time.time()
        history, outputs_before, outputs_after = train_one_lr(
            lr=lr,
            initial_state=initial_state,
            batch=batch,
            y_isp=y_isp,
            uv_isp=uv_isp,
            pattern=pattern,
            weights=weights,
            composite_cfg=composite_cfg,
            steps=args.steps,
            device=device,
        )
        elapsed = time.time() - t0

        summary = summarize_run(history)
        summary["elapsed_s"] = elapsed
        summaries.append(summary)
        all_rows.extend(history)

        start = summary["start"]
        final = summary["final"]
        print("\nSummary:")
        print(f"  loss:      {start['loss']:.6f} -> {final['loss']:.6f}")
        print(f"  MS-SSIM:   {start['ms_ssim']:.4f} -> {final['ms_ssim']:.4f}")
        print(f"  VIF:       {start['vif']:.4f} -> {final['vif']:.4f}")
        print(f"  UNIQUE:    {start['unique']:.4f} -> {final['unique']:.4f} raw")
        print(f"  UNIQUE_n:  {start['unique_norm']:.4f} -> {final['unique_norm']:.4f}")
        print(f"  L1_UV:     {start['l1_uv']:.6f} -> {final['l1_uv']:.6f}")
        print(f"  composite: {start['composite']:.4f} -> {final['composite']:.4f}")
        print(f"  CNN grads: {'OK' if summary['cnn_grads_present'] else 'MISSING'}")
        print(f"  elapsed:   {elapsed:.1f}s")

        if best_final is None or final["composite"] > best_final["composite"]:
            best_final = final
            best_outputs = (outputs_before, outputs_after)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "quality_overfit_history.csv"
    save_history_csv(csv_path, all_rows)
    print(f"\nHistory saved to {csv_path}")

    summary_path = output_dir / "quality_overfit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to {summary_path}")

    if best_outputs is not None:
        save_visualization(output_dir, batch, best_outputs[0], best_outputs[1])

    print("\nQuality overfit test complete.")


if __name__ == "__main__":
    main()
