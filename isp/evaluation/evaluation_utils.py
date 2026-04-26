import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import pyiqa
except ModuleNotFoundError:
    pyiqa = None

try:
    from isp.color.conversions import (
        float_y_to_uint8,
        normalize_uv_planes,
        normalize_y_plane,
        unpack_nv12_buffer,
        yuv420_to_rgb_bt709_full,
        yuv420_to_yuv444,
        yuv444_to_yuv420,
    )
    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader
    from metrics.vif import vif_cfa_to_y
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from isp.color.conversions import (
        float_y_to_uint8,
        normalize_uv_planes,
        normalize_y_plane,
        unpack_nv12_buffer,
        yuv420_to_rgb_bt709_full,
        yuv420_to_yuv444,
        yuv444_to_yuv420,
    )
    from isp.config.config_reader import read_config
    from isp.io.video_reader import RAWVideoReader
    from isp.io.yuv_reader import NV12VideoReader
    from metrics.vif import vif_cfa_to_y


ROOT = Path(__file__).resolve().parents[2]


def load_split_items(splits_json_path: str, split_name: str) -> list[dict[str, Any]]:
    """
    Load eval items for a split and resolve relative paths to project-root paths.
    """
    splits_path = Path(splits_json_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Split file not found: {splits_path}")

    with open(splits_path, encoding="utf-8") as f:
        payload = json.load(f)

    split_payload = payload.get("splits", {})
    if split_name not in split_payload:
        available = ", ".join(sorted(split_payload.keys()))
        raise KeyError(f"Unknown split '{split_name}'. Available: {available}")

    items: list[dict[str, Any]] = []
    for item in split_payload[split_name]:
        raw_path = Path(item["raw_path"])
        yuv_path = Path(item["yuv_path"])

        if not raw_path.is_absolute():
            raw_path = (ROOT / raw_path).resolve()
        if not yuv_path.is_absolute():
            yuv_path = (ROOT / yuv_path).resolve()

        items.append(
            {
                "scene": item["scene"],
                "scene_id": int(item["scene_id"]),
                "raw_path": str(raw_path),
                "yuv_path": str(yuv_path),
                "frame_indices": [int(frame_idx) for frame_idx in item["frame_indices"]],
                "frame_count": int(item.get("frame_count", len(item["frame_indices"]))),
            }
        )

    return items


def limit_eval_items(
    eval_items: list[dict[str, Any]], max_frames: int | None
) -> list[dict[str, Any]]:
    """
    Keep only the first max_frames frames across all eval items.
    """
    items = deepcopy(eval_items)

    if max_frames is None:
        return items
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    limited_items: list[dict[str, Any]] = []
    remaining = max_frames

    for item in items:
        if remaining <= 0:
            break

        frame_indices = item["frame_indices"][:remaining]
        if not frame_indices:
            continue

        item["frame_indices"] = frame_indices
        item["frame_count"] = len(frame_indices)
        limited_items.append(item)
        remaining -= len(frame_indices)

    return limited_items


def _coerce_y_tensor(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        if y.shape[0] == 1:
            y = y.unsqueeze(0)
        else:
            y = y.unsqueeze(1)
    elif y.ndim != 4:
        raise ValueError(f"Unsupported Y tensor shape: {tuple(y.shape)}")

    if y.shape[1] != 1:
        raise ValueError(f"Expected one-channel Y tensor, got {tuple(y.shape)}")

    return y


def _coerce_uv_tensor(uv: torch.Tensor) -> torch.Tensor:
    if uv.ndim == 3:
        uv = uv.unsqueeze(0)
    elif uv.ndim != 4:
        raise ValueError(f"Unsupported UV tensor shape: {tuple(uv.shape)}")

    if uv.shape[1] != 2:
        raise ValueError(f"Expected two-channel UV tensor, got {tuple(uv.shape)}")

    return uv


def run_isp_frame(
    isp, raw_frame: torch.Tensor, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run ISP on one RAW frame and return planar Y and UV in float32 [0, 1].
    """
    if hasattr(isp, "forward_components"):
        components = isp.forward_components(raw_frame)

        if isinstance(components, dict):
            if "nv12" in components:
                y_u8, u_u8, v_u8 = unpack_nv12_buffer(components["nv12"], width, height)
                y = normalize_y_plane(y_u8)
                uv = normalize_uv_planes(u_u8, v_u8)
            elif "y" in components and "uv" in components:
                y = _coerce_y_tensor(components["y"])
                uv = _coerce_uv_tensor(components["uv"])
            elif "y" in components and "u" in components and "v" in components:
                y = _coerce_y_tensor(components["y"])
                uv = normalize_uv_planes(
                    _coerce_y_tensor(components["u"]),
                    _coerce_y_tensor(components["v"]),
                )
            else:
                raise ValueError("forward_components() returned unsupported dict keys")
        elif isinstance(components, (tuple, list)):
            if len(components) == 2:
                y = _coerce_y_tensor(components[0])
                uv = _coerce_uv_tensor(components[1])
            elif len(components) == 3:
                y = _coerce_y_tensor(components[0])
                uv = normalize_uv_planes(
                    _coerce_y_tensor(components[1]),
                    _coerce_y_tensor(components[2]),
                )
            else:
                raise ValueError("forward_components() returned unsupported tuple length")
        else:
            raise TypeError("forward_components() must return dict, tuple, or list")

        y = y.to(torch.float32)
        uv = uv.to(torch.float32)

        if y.max().item() > 1.0 or uv.max().item() > 1.0:
            y = y / 255.0 if y.max().item() > 1.0 else y
            uv = uv / 255.0 if uv.max().item() > 1.0 else uv

        return y.clamp(0.0, 1.0), uv.clamp(0.0, 1.0)

    nv12 = isp(raw_frame)
    y_u8, u_u8, v_u8 = unpack_nv12_buffer(nv12, width, height)

    y = normalize_y_plane(y_u8)
    uv = normalize_uv_planes(u_u8, v_u8)

    return y, uv


def run_model_frame(
    model, y_isp: torch.Tensor, uv_isp: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run residual CNN on YUV444 and return planar YUV420 prediction.
    """
    if model is None:
        return y_isp, uv_isp

    yuv444_isp = yuv420_to_yuv444(y_isp, uv_isp)
    residual = model(yuv444_isp)

    if not isinstance(residual, torch.Tensor):
        raise TypeError(f"Model must return torch.Tensor, got {type(residual)!r}")
    if residual.shape != yuv444_isp.shape:
        raise ValueError(
            f"Model residual shape {tuple(residual.shape)} does not match "
            f"input {tuple(yuv444_isp.shape)}"
        )

    yuv444_pred = torch.clamp(yuv444_isp + residual, 0.0, 1.0)
    return yuv444_to_yuv420(yuv444_pred)


def compute_vif_from_raw_and_y(
    raw_frame: torch.Tensor, y_pred: torch.Tensor, pattern: str
) -> torch.Tensor:
    """
    Compute VIF between RAW CFA and predicted Y plane.
    """
    if raw_frame.ndim == 2:
        cfa = raw_frame.unsqueeze(0).unsqueeze(0)
    elif raw_frame.ndim == 4 and raw_frame.shape[1] == 1:
        cfa = raw_frame
    else:
        raise ValueError(f"Unexpected RAW frame shape: {tuple(raw_frame.shape)}")

    cfa_16bit = (cfa.to(torch.float32) * (65535.0 / 4095.0)).clamp(0.0, 65535.0).to(torch.int32)
    y_uint8 = float_y_to_uint8(y_pred)
    return vif_cfa_to_y(cfa=cfa_16bit, y=y_uint8, pattern=pattern, even=False)


def init_iqa_metrics(device: str):
    """
    Initialize pyiqa metrics for no-reference evaluation.
    """
    if pyiqa is None:
        raise ModuleNotFoundError("pyiqa is not available, cannot compute NRQM/UNIQUE")

    device_obj = torch.device(device)
    nrqm_metric = pyiqa.create_metric("nrqm", device=device_obj)
    unique_metric = pyiqa.create_metric("unique", device=device_obj)
    return nrqm_metric, unique_metric


def init_metric_sums(compute_iqa: bool) -> dict[str, Any]:
    """
    Create metric accumulators for total and per-scene averages.
    """
    return {
        "total": {
            "count": 0,
            "l1_y": 0.0,
            "l1_uv": 0.0,
            "vif": 0.0,
            "nrqm": 0.0 if compute_iqa else None,
            "unique": 0.0 if compute_iqa else None,
        },
        "per_scene": {},
        "compute_iqa": compute_iqa,
    }


def update_metric_sums(
    metric_sums: dict[str, Any],
    scene_name: str,
    l1_y: float,
    l1_uv: float,
    vif: float,
    nrqm: float | None,
    unique: float | None,
):
    """
    Add one frame worth of metric values to total and per-scene sums.
    """
    compute_iqa = metric_sums["compute_iqa"]

    if scene_name not in metric_sums["per_scene"]:
        metric_sums["per_scene"][scene_name] = {
            "count": 0,
            "l1_y": 0.0,
            "l1_uv": 0.0,
            "vif": 0.0,
            "nrqm": 0.0 if compute_iqa else None,
            "unique": 0.0 if compute_iqa else None,
        }

    for scope in [metric_sums["total"], metric_sums["per_scene"][scene_name]]:
        scope["count"] += 1
        scope["l1_y"] += l1_y
        scope["l1_uv"] += l1_uv
        scope["vif"] += vif
        if compute_iqa:
            scope["nrqm"] += float(nrqm)
            scope["unique"] += float(unique)


def finalize_metric_sums(metric_sums: dict[str, Any]) -> dict[str, Any]:
    """
    Convert metric sums into averages.
    """
    compute_iqa = metric_sums["compute_iqa"]

    def finalize_scope(scope: dict[str, Any]) -> dict[str, Any]:
        count = int(scope["count"])
        if count <= 0:
            return {
                "l1_y": None,
                "l1_uv": None,
                "vif": None,
                "nrqm": None,
                "unique": None,
                "num_frames": 0,
            }

        result = {
            "l1_y": scope["l1_y"] / count,
            "l1_uv": scope["l1_uv"] / count,
            "vif": scope["vif"] / count,
            "nrqm": scope["nrqm"] / count if compute_iqa else None,
            "unique": scope["unique"] / count if compute_iqa else None,
            "num_frames": count,
        }
        return result

    total_result = finalize_scope(metric_sums["total"])
    per_scene_result = {
        scene_name: finalize_scope(scene_scope)
        for scene_name, scene_scope in metric_sums["per_scene"].items()
    }

    total_result["per_scene"] = per_scene_result
    return total_result


def evaluate(
    isp,
    model,
    eval_items: list[dict[str, Any]],
    config_path: str,
    device: str = "cuda",
    compute_iqa: bool = True,
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate ISP or ISP+CNN on selected frames from one or more scenes.
    """
    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        config_path_obj = (ROOT / config_path_obj).resolve()

    eval_items = limit_eval_items(eval_items, max_frames)
    if not eval_items:
        raise ValueError("No frames selected for evaluation")

    config = read_config(str(config_path_obj), device=device)
    width = int(config["img"]["width"])
    height = int(config["img"]["height"])
    top_lines, bottom_lines = config["img"]["emb_lines"]
    out_height = int(height - top_lines - bottom_lines)
    pattern = str(config["img"].get("bayer_pattern", "RGGB"))

    if compute_iqa:
        nrqm_metric, unique_metric = init_iqa_metrics(device)
    else:
        nrqm_metric = None
        unique_metric = None

    metric_sums = init_metric_sums(compute_iqa)
    total_frames = sum(len(item["frame_indices"]) for item in eval_items)

    isp_was_training = getattr(isp, "training", None)
    model_was_training = getattr(model, "training", None) if model is not None else None

    if hasattr(isp, "eval"):
        isp.eval()
    if model is not None and hasattr(model, "eval"):
        model.eval()

    pbar = tqdm(total=total_frames, desc="Evaluate", disable=not verbose)

    try:
        with torch.no_grad():
            for item in eval_items:
                scene_name = item["scene"]
                raw_path = Path(item["raw_path"])
                yuv_path = Path(item["yuv_path"])
                frame_indices = item["frame_indices"]

                selected_set = set(frame_indices)
                expected_frames = len(frame_indices)
                processed_frames = 0

                with (
                    RAWVideoReader(str(raw_path), config, device=device) as raw_reader,
                    NV12VideoReader(str(yuv_path), width, out_height, device=device) as yuv_reader,
                ):
                    for (raw_frame, raw_number), (yuv_frame, yuv_number) in zip(
                        raw_reader, yuv_reader, strict=False
                    ):
                        if raw_number != yuv_number:
                            raise RuntimeError(
                                f"RAW/YUV frame mismatch in scene '{scene_name}': "
                                f"{raw_number} vs {yuv_number}"
                            )

                        frame_idx = raw_number - 1
                        if frame_idx not in selected_set:
                            continue

                        y_ref_u8, u_ref_u8, v_ref_u8 = yuv_frame
                        y_ref = normalize_y_plane(y_ref_u8)
                        uv_ref = normalize_uv_planes(u_ref_u8, v_ref_u8)

                        y_isp, uv_isp = run_isp_frame(isp, raw_frame, width, out_height)
                        y_pred, uv_pred = run_model_frame(model, y_isp, uv_isp)

                        l1_y = F.l1_loss(y_pred, y_ref).item()
                        l1_uv = F.l1_loss(uv_pred, uv_ref).item()
                        vif_value = compute_vif_from_raw_and_y(raw_frame, y_pred, pattern).item()

                        if compute_iqa:
                            rgb_pred = yuv420_to_rgb_bt709_full(y_pred, uv_pred)
                            nrqm_value = float(nrqm_metric(rgb_pred).item())
                            unique_value = float(unique_metric(rgb_pred).item())
                        else:
                            nrqm_value = None
                            unique_value = None

                        update_metric_sums(
                            metric_sums,
                            scene_name=scene_name,
                            l1_y=l1_y,
                            l1_uv=l1_uv,
                            vif=vif_value,
                            nrqm=nrqm_value,
                            unique=unique_value,
                        )

                        processed_frames += 1
                        pbar.update(1)

                        postfix = {
                            "scene": scene_name,
                            "frame": frame_idx,
                            "l1_y": f"{l1_y:.4f}",
                            "vif": f"{vif_value:.4f}",
                        }
                        if compute_iqa:
                            postfix["nrqm"] = f"{nrqm_value:.4f}"
                            postfix["unique"] = f"{unique_value:.4f}"
                        pbar.set_postfix(postfix)

                        if processed_frames == expected_frames:
                            break

                if processed_frames != expected_frames:
                    raise RuntimeError(
                        f"Scene '{scene_name}' expected {expected_frames} frames, "
                        f"processed {processed_frames}"
                    )
    finally:
        pbar.close()
        if isp_was_training is True and hasattr(isp, "train"):
            isp.train()
        if model is not None and model_was_training is True and hasattr(model, "train"):
            model.train()

    return finalize_metric_sums(metric_sums)


def evaluate_split(
    isp,
    model,
    splits_json_path: str,
    split_name: str,
    config_path: str,
    device: str = "cuda",
    compute_iqa: bool = True,
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convenience wrapper: load eval items from splits.json and evaluate them.
    """
    eval_items = load_split_items(splits_json_path, split_name)
    return evaluate(
        isp=isp,
        model=model,
        eval_items=eval_items,
        config_path=config_path,
        device=device,
        compute_iqa=compute_iqa,
        max_frames=max_frames,
        verbose=verbose,
    )
