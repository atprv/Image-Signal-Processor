"""
Render full-scene YUV videos from a checkpoint pipeline and evaluate them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isp.color.conversions import (
    float_y_to_uint8,
    normalize_uv_planes,
    normalize_y_plane,
    yuv420_to_rgb_bt709_full,
)
from isp.config.config_reader import read_config
from isp.evaluation.composite_score import compute_composite, load_composite_config
from isp.evaluation.evaluation_utils import (
    compute_vif_from_raw_and_y,
    init_iqa_metrics,
    run_isp_frame,
    run_model_frame,
)
from isp.io.video_reader import RAWVideoReader
from isp.io.video_writer import AsyncYUVWriter
from isp.io.yuv_reader import NV12VideoReader
from isp.pipeline.pipeline import ISPPipeline
from scripts.run_optuna_isp_knobs import (
    diff_only_isp_state,
    load_cnn,
    sanitize_loaded_isp_state,
    trained_param_keys,
)

SCENE_FILES: dict[str, dict[str, str]] = {
    "day": {
        "raw_path": "data/day_0.bin",
        "ref_yuv_path": "data/day_0.yuv",
    },
    "night": {
        "raw_path": "data/night_0.bin",
        "ref_yuv_path": "data/night_0.yuv",
    },
    "tunnel": {
        "raw_path": "data/tunnel_0.bin",
        "ref_yuv_path": "data/tunnel_0.yuv",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render full-scene YUV videos from a checkpoint and evaluate them."
    )
    p.add_argument(
        "--ckpt",
        default="artifacts/checkpoints/e2e_quality/e2e_best.pth",
        help="Checkpoint with isp_state_dict and cnn_state_dict.",
    )
    p.add_argument(
        "--optuna-best-json",
        default="artifacts/checkpoints/optuna_tuning/optuna_best_params.json",
        help="Optional Optuna JSON with the best structural ISP knobs.",
    )
    p.add_argument("--config", default="data/imx623.toml")
    p.add_argument(
        "--norm-weights",
        default="artifacts/baselines/norm_weights.json",
        help="Frozen composite-score weights used in reports when IQA is enabled.",
    )
    p.add_argument(
        "--mode",
        choices=["isp", "isp_cnn"],
        default="isp_cnn",
        help="Render ISP-only output or the full ISP+CNN pipeline.",
    )
    p.add_argument(
        "--scenes",
        default="day,night,tunnel",
        help="Comma-separated scene list. Supported: day, night, tunnel.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device for rendering and metrics.",
    )
    p.add_argument(
        "--metric-stride",
        type=int,
        default=1,
        help="Evaluate every N-th frame from the rendered YUV. 1 = all frames.",
    )
    p.add_argument(
        "--no-iqa",
        action="store_true",
        help="Skip NRQM/UNIQUE to speed up evaluation.",
    )
    p.add_argument(
        "--skip-render",
        action="store_true",
        help="Do not render videos; only evaluate existing outputs.",
    )
    p.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render outputs even if the target YUV file already exists.",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/results/full_video_eval",
        help="Output directory for YUVs and reports.",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_scene_list(scene_csv: str) -> list[str]:
    scenes = [scene.strip() for scene in scene_csv.split(",") if scene.strip()]
    if not scenes:
        raise ValueError("At least one scene must be selected.")
    unsupported = [scene for scene in scenes if scene not in SCENE_FILES]
    if unsupported:
        raise KeyError(f"Unsupported scene(s): {unsupported}. Supported: {sorted(SCENE_FILES)}")
    return scenes


def load_optuna_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    json_path = resolve(path)
    if not json_path.exists():
        print(f"Optuna JSON not found, continuing without structural overrides: {json_path}")
        return {}

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    overrides = payload.get("construction_kwargs") or payload.get("best_params") or {}
    if not isinstance(overrides, dict):
        raise TypeError(f"Expected dict in Optuna JSON, got {type(overrides)!r}")
    return dict(overrides)


def build_trained_isp(
    payload: dict[str, Any],
    config: dict[str, Any],
    device: str,
    structural_overrides: dict[str, Any],
) -> ISPPipeline:
    """Rebuild ISP from checkpoint and optional structural Optuna overrides."""
    isp = ISPPipeline(config, device=device, **structural_overrides)
    param_keys = trained_param_keys(isp)
    trained_state = diff_only_isp_state(payload, param_keys)
    trained_state = sanitize_loaded_isp_state(isp, trained_state, structural_overrides)
    _, unexpected = isp.load_state_dict(trained_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected ISP keys while loading checkpoint: {unexpected}")
    isp.eval()
    for param in isp.parameters():
        param.requires_grad_(False)
    return isp


def pack_yuv420_to_nv12(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Pack float Y / UV tensors into a flat NV12 uint8 buffer."""
    if y.ndim != 4 or y.shape[0] != 1 or y.shape[1] != 1:
        raise ValueError(f"Expected Y tensor [1,1,H,W], got {tuple(y.shape)}")
    if uv.ndim != 4 or uv.shape[0] != 1 or uv.shape[1] != 2:
        raise ValueError(f"Expected UV tensor [1,2,H/2,W/2], got {tuple(uv.shape)}")

    y_u8 = float_y_to_uint8(y).reshape(-1)
    uv_u8 = (uv * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)[0]
    uv_interleaved = uv_u8.permute(1, 2, 0).contiguous().reshape(-1)
    return torch.cat([y_u8, uv_interleaved], dim=0)


def render_scene_video(
    scene_name: str,
    scene_cfg: dict[str, str],
    config: dict[str, Any],
    config_path: Path,
    payload: dict[str, Any],
    device: str,
    mode: str,
    structural_overrides: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Render one full-scene YUV file from the checkpoint pipeline."""
    del config_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    isp = build_trained_isp(payload, config, device, structural_overrides)
    model = load_cnn(payload, device) if mode == "isp_cnn" else None

    width = int(config["img"]["width"])
    height = int(config["img"]["height"])
    top_lines, bottom_lines = config["img"]["emb_lines"]
    out_height = int(height - top_lines - bottom_lines)

    raw_path = resolve(scene_cfg["raw_path"])
    frame_count = 0
    start_time = time.perf_counter()

    with (
        RAWVideoReader(str(raw_path), config, device=device) as raw_reader,
        AsyncYUVWriter(str(output_path)) as writer,
    ):
        with torch.no_grad():
            pbar = tqdm(desc=f"Render {scene_name}", unit="frame")
            try:
                for raw_frame, frame_number in raw_reader:
                    if mode == "isp":
                        nv12 = isp(raw_frame)
                    else:
                        y_isp, uv_isp = run_isp_frame(isp, raw_frame, width, out_height)
                        y_pred, uv_pred = run_model_frame(model, y_isp, uv_isp)
                        nv12 = pack_yuv420_to_nv12(y_pred, uv_pred)

                    writer.write(nv12.cpu())
                    frame_count += 1
                    pbar.update(1)
                    if frame_count % 50 == 0:
                        elapsed = time.perf_counter() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0.0
                        pbar.set_postfix({"frame": frame_number - 1, "fps": f"{fps:.2f}"})
            finally:
                pbar.close()

    elapsed = time.perf_counter() - start_time
    return {
        "scene": scene_name,
        "output_yuv": str(output_path),
        "render_num_frames": frame_count,
        "render_seconds": elapsed,
        "render_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "mode": mode,
    }


def evaluate_rendered_scene(
    scene_name: str,
    scene_cfg: dict[str, str],
    config: dict[str, Any],
    pred_yuv_path: Path,
    device: str,
    compute_iqa: bool,
    metric_stride: int,
    composite_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one rendered NV12 video against the reference YUV."""
    if metric_stride <= 0:
        raise ValueError(f"metric_stride must be positive, got {metric_stride}")

    width = int(config["img"]["width"])
    height = int(config["img"]["height"])
    top_lines, bottom_lines = config["img"]["emb_lines"]
    out_height = int(height - top_lines - bottom_lines)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    raw_path = resolve(scene_cfg["raw_path"])
    ref_yuv_path = resolve(scene_cfg["ref_yuv_path"])

    nrqm_metric = None
    unique_metric = None
    if compute_iqa:
        nrqm_metric, unique_metric = init_iqa_metrics(device)

    sums = {
        "count": 0,
        "l1_y": 0.0,
        "l1_uv": 0.0,
        "vif": 0.0,
        "nrqm": 0.0 if compute_iqa else None,
        "unique": 0.0 if compute_iqa else None,
    }

    start_time = time.perf_counter()
    with (
        RAWVideoReader(str(raw_path), config, device=device) as raw_reader,
        NV12VideoReader(str(ref_yuv_path), width, out_height, device=device) as ref_reader,
        NV12VideoReader(str(pred_yuv_path), width, out_height, device=device) as pred_reader,
    ):
        pbar = tqdm(desc=f"Metric {scene_name}", unit="frame")
        try:
            for raw_pack, ref_pack, pred_pack in zip(
                raw_reader, ref_reader, pred_reader, strict=True
            ):
                (raw_frame, raw_number) = raw_pack
                ((y_ref_u8, u_ref_u8, v_ref_u8), ref_number) = ref_pack
                ((y_pred_u8, u_pred_u8, v_pred_u8), pred_number) = pred_pack

                if raw_number != ref_number or raw_number != pred_number:
                    raise RuntimeError(
                        f"Frame mismatch in scene {scene_name}: "
                        f"raw={raw_number}, ref={ref_number}, pred={pred_number}"
                    )

                frame_idx = raw_number - 1
                pbar.update(1)
                if frame_idx % metric_stride != 0:
                    continue

                y_ref = normalize_y_plane(y_ref_u8)
                uv_ref = normalize_uv_planes(u_ref_u8, v_ref_u8)
                y_pred = normalize_y_plane(y_pred_u8)
                uv_pred = normalize_uv_planes(u_pred_u8, v_pred_u8)

                l1_y = float(torch.nn.functional.l1_loss(y_pred, y_ref).item())
                l1_uv = float(torch.nn.functional.l1_loss(uv_pred, uv_ref).item())
                vif_value = float(compute_vif_from_raw_and_y(raw_frame, y_pred, pattern).item())

                sums["count"] += 1
                sums["l1_y"] += l1_y
                sums["l1_uv"] += l1_uv
                sums["vif"] += vif_value

                if compute_iqa:
                    rgb_pred = yuv420_to_rgb_bt709_full(y_pred, uv_pred)
                    nrqm_value = float(nrqm_metric(rgb_pred).item())
                    unique_value = float(unique_metric(rgb_pred).item())
                    sums["nrqm"] += nrqm_value
                    sums["unique"] += unique_value
                    pbar.set_postfix(
                        {
                            "frame": frame_idx,
                            "vif": f"{vif_value:.4f}",
                            "unique": f"{unique_value:.4f}",
                        }
                    )
                else:
                    pbar.set_postfix({"frame": frame_idx, "vif": f"{vif_value:.4f}"})
        finally:
            pbar.close()

    if sums["count"] <= 0:
        raise RuntimeError(f"No frames evaluated for scene '{scene_name}'.")

    result = {
        "scene": scene_name,
        "num_frames": int(sums["count"]),
        "metric_stride": int(metric_stride),
        "l1_y": sums["l1_y"] / sums["count"],
        "l1_uv": sums["l1_uv"] / sums["count"],
        "vif": sums["vif"] / sums["count"],
        "nrqm": (sums["nrqm"] / sums["count"]) if compute_iqa else None,
        "unique": (sums["unique"] / sums["count"]) if compute_iqa else None,
        "metric_seconds": time.perf_counter() - start_time,
    }
    if compute_iqa and composite_cfg is not None:
        result["composite"] = compute_composite(
            result["vif"], result["nrqm"], result["unique"], composite_cfg
        )
    else:
        result["composite"] = None
    return result


def aggregate_scene_reports(scene_reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scene_reports = list(scene_reports)
    total_frames = sum(int(report["num_frames"]) for report in scene_reports)
    if total_frames <= 0:
        raise ValueError("Cannot aggregate reports with zero frames.")

    def weighted(metric: str) -> float | None:
        values = [report.get(metric) for report in scene_reports]
        if any(value is None for value in values):
            return None
        return (
            sum(float(report[metric]) * int(report["num_frames"]) for report in scene_reports)
            / total_frames
        )

    return {
        "num_frames": total_frames,
        "l1_y": weighted("l1_y"),
        "l1_uv": weighted("l1_uv"),
        "vif": weighted("vif"),
        "nrqm": weighted("nrqm"),
        "unique": weighted("unique"),
        "composite": weighted("composite"),
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Full video render + metric report")
    lines.append("")
    lines.append("## Run setup")
    lines.append(f"- mode: `{report['mode']}`")
    lines.append(f"- checkpoint: `{report['checkpoint']}`")
    lines.append(f"- optuna overrides: `{report['optuna_best_json']}`")
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- scenes: `{', '.join(report['scenes'])}`")
    lines.append(f"- metric_stride: `{report['metric_stride']}`")
    lines.append(f"- compute_iqa: `{report['compute_iqa']}`")
    lines.append("")
    lines.append("## Overall")
    overall = report["overall_metrics"]
    lines.append(f"- frames: `{overall['num_frames']}`")
    lines.append(f"- L1_Y: `{overall['l1_y']:.6f}`")
    lines.append(f"- L1_UV: `{overall['l1_uv']:.6f}`")
    lines.append(f"- VIF: `{overall['vif']:.6f}`")
    if overall["nrqm"] is not None:
        lines.append(f"- NRQM: `{overall['nrqm']:.6f}`")
        lines.append(f"- UNIQUE: `{overall['unique']:.6f}`")
        lines.append(f"- Composite: `{overall['composite']:.6f}`")
    lines.append("")
    lines.append("## Per Scene")
    for scene_name in report["scenes"]:
        metrics = report["scene_metrics"][scene_name]
        render = report["scene_render"][scene_name]
        lines.append(f"### {scene_name}")
        lines.append(f"- output: `{render['output_yuv']}`")
        lines.append(f"- rendered frames: `{render['render_num_frames']}`")
        lines.append(f"- render FPS: `{render['render_fps']:.3f}`")
        lines.append(f"- eval frames: `{metrics['num_frames']}`")
        lines.append(f"- L1_Y: `{metrics['l1_y']:.6f}`")
        lines.append(f"- L1_UV: `{metrics['l1_uv']:.6f}`")
        lines.append(f"- VIF: `{metrics['vif']:.6f}`")
        if metrics["nrqm"] is not None:
            lines.append(f"- NRQM: `{metrics['nrqm']:.6f}`")
            lines.append(f"- UNIQUE: `{metrics['unique']:.6f}`")
            lines.append(f"- Composite: `{metrics['composite']:.6f}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    config_path = resolve(args.config)
    ckpt_path = resolve(args.ckpt)
    optuna_json_path = resolve(args.optuna_best_json) if args.optuna_best_json else None
    norm_weights_path = resolve(args.norm_weights)
    output_dir = resolve(args.output_dir)
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    scenes = parse_scene_list(args.scenes)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = read_config(str(config_path), device=device)
    structural_overrides = load_optuna_overrides(args.optuna_best_json)
    composite_cfg = None if args.no_iqa else load_composite_config(norm_weights_path)

    print("=" * 72)
    print("Full checkpoint render + metric evaluation")
    print("=" * 72)
    print(f"Mode:            {args.mode}")
    print(f"Device:          {device}")
    print(f"Checkpoint:      {ckpt_path}")
    print(f"Optuna overrides:{optuna_json_path}")
    print(f"Scenes:          {scenes}")
    print(f"Metric stride:   {args.metric_stride}")
    print(f"Compute IQA:     {not args.no_iqa}")
    print(f"Output dir:      {output_dir}")
    if structural_overrides:
        print(f"Loaded {len(structural_overrides)} structural override(s) from Optuna JSON.")
    print()

    scene_render: dict[str, dict[str, Any]] = {}
    scene_metrics: dict[str, dict[str, Any]] = {}

    for scene_name in scenes:
        scene_cfg = SCENE_FILES[scene_name]
        pred_yuv_path = videos_dir / f"{scene_name}_{args.mode}.yuv"

        if args.skip_render:
            if not pred_yuv_path.exists():
                raise FileNotFoundError(
                    f"--skip-render was requested but output is missing: {pred_yuv_path}"
                )
            scene_render[scene_name] = {
                "scene": scene_name,
                "output_yuv": str(pred_yuv_path),
                "render_num_frames": None,
                "render_seconds": None,
                "render_fps": None,
                "mode": args.mode,
            }
        elif pred_yuv_path.exists() and not args.force_render:
            print(f"Reuse existing render: {pred_yuv_path}")
            scene_render[scene_name] = {
                "scene": scene_name,
                "output_yuv": str(pred_yuv_path),
                "render_num_frames": None,
                "render_seconds": None,
                "render_fps": None,
                "mode": args.mode,
            }
        else:
            print(f"Rendering scene '{scene_name}' -> {pred_yuv_path.name}")
            scene_render[scene_name] = render_scene_video(
                scene_name=scene_name,
                scene_cfg=scene_cfg,
                config=config,
                config_path=config_path,
                payload=payload,
                device=device,
                mode=args.mode,
                structural_overrides=structural_overrides,
                output_path=pred_yuv_path,
            )

        print(f"Evaluating scene '{scene_name}' from rendered YUV")
        scene_metrics[scene_name] = evaluate_rendered_scene(
            scene_name=scene_name,
            scene_cfg=scene_cfg,
            config=config,
            pred_yuv_path=pred_yuv_path,
            device=device,
            compute_iqa=not args.no_iqa,
            metric_stride=args.metric_stride,
            composite_cfg=composite_cfg,
        )

    overall_metrics = aggregate_scene_reports(scene_metrics.values())

    report = {
        "checkpoint": str(ckpt_path),
        "optuna_best_json": str(optuna_json_path) if optuna_json_path else None,
        "mode": args.mode,
        "device": device,
        "scenes": scenes,
        "metric_stride": int(args.metric_stride),
        "compute_iqa": bool(not args.no_iqa),
        "structural_overrides": structural_overrides,
        "scene_render": scene_render,
        "scene_metrics": scene_metrics,
        "overall_metrics": overall_metrics,
    }

    json_path = output_dir / f"report_{args.mode}.json"
    md_path = output_dir / f"report_{args.mode}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown_report(md_path, report)

    print("\nOverall metrics:")
    print(f"  frames:   {overall_metrics['num_frames']}")
    print(f"  L1_Y:     {overall_metrics['l1_y']:.6f}")
    print(f"  L1_UV:    {overall_metrics['l1_uv']:.6f}")
    print(f"  VIF:      {overall_metrics['vif']:.6f}")
    if overall_metrics["nrqm"] is not None:
        print(f"  NRQM:     {overall_metrics['nrqm']:.6f}")
        print(f"  UNIQUE:   {overall_metrics['unique']:.6f}")
        print(f"  Composite:{overall_metrics['composite']:.6f}")

    print(f"\nSaved JSON report: {json_path}")
    print(f"Saved MD report:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
