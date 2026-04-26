"""
Quality-aligned differentiable loss for E2E ISP training.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from isp.color.conversions import yuv420_to_rgb_bt709_full

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    import pyiqa
except ModuleNotFoundError:
    pyiqa = None

try:
    from metrics.vif import vif_cfa_to_y
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from metrics.vif import vif_cfa_to_y


@dataclass
class QualityLossWeights:
    """Weights for the quality-aligned loss. Positive = stronger pressure."""

    w_ssim: float = 1.0
    w_vif: float = 0.3
    w_unique: float = 0.1
    w_l1_y: float = 0.0
    w_uv: float = 1.0


def _weighted_mean(
    values: torch.Tensor, sample_weights: torch.Tensor | None = None
) -> torch.Tensor:
    if values.ndim == 0:
        return values
    values = values.reshape(values.shape[0], -1).mean(dim=1)
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if weights.numel() != values.numel():
        raise ValueError(
            f"sample_weights length {weights.numel()} does not match batch size {values.numel()}"
        )
    weights = weights / weights.sum().clamp_min(1e-8)
    return (values * weights).sum()


_METRIC_CACHE: dict[str, dict[str, object]] = {}


def _get_pyiqa_metric(name: str, device: torch.device):
    """
    Return a cached pyiqa metric instance for (name, device).
    """
    if pyiqa is None:
        raise ModuleNotFoundError(
            "pyiqa is required for quality_loss (MS_SSIM / UNIQUE). "
            "Install via `pip install pyiqa`."
        )
    dev_key = str(device)
    bucket = _METRIC_CACHE.setdefault(dev_key, {})
    if name not in bucket:
        bucket[name] = pyiqa.create_metric(name, device=device, as_loss=True)
    return bucket[name]


def compute_vif_from_raw_and_y_diff(
    raw_12bit: torch.Tensor, y_pred: torch.Tensor, pattern: str
) -> torch.Tensor:
    """
    Differentiable variant of compute_vif_from_raw_and_y().

    Args:
        raw_12bit: [B, 1, H, W] float in [0, 4095]
        y_pred:    [B, 1, H, W] float in [0, 1]
        pattern:   CFA pattern string

    Returns:
        vif_per_batch: [B] tensor of per-sample VIF values in [0, 1].
    """
    if raw_12bit.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Batch size mismatch: raw_12bit[B={raw_12bit.shape[0]}] vs y_pred[B={y_pred.shape[0]}]"
        )

    cfa_scaled = (raw_12bit * (65535.0 / 4095.0)).clamp(0.0, 65535.0)
    y_scaled = y_pred.clamp(0.0, 1.0) * 255.0

    per_sample = []
    for i in range(cfa_scaled.shape[0]):
        vi = vif_cfa_to_y(
            cfa=cfa_scaled[i : i + 1],
            y=y_scaled[i : i + 1],
            pattern=pattern,
            even=False,
        )
        per_sample.append(vi)
    return torch.cat(per_sample, dim=0)


def compute_quality_loss(
    raw_batch: torch.Tensor,
    y_pred: torch.Tensor,
    uv_pred: torch.Tensor,
    y_ref: torch.Tensor,
    uv_ref: torch.Tensor,
    pattern: str,
    weights: QualityLossWeights | None = None,
    lambda_uv: float | None = None,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Differentiable quality-aligned loss. Signature mirrors compute_proxy_loss
    so it can be dropped into e2e_train_step.

    Args:
        raw_batch:  [B, 1, H, W] RAW CFA (accepted in [0,1] or [0,4095])
        y_pred:     [B, 1, H, W] float in [0, 1]
        uv_pred:    [B, 2, H/2, W/2] float in [0, 1]
        y_ref:      [B, 1, H, W] float in [0, 1]
        uv_ref:     [B, 2, H/2, W/2] float in [0, 1]
        pattern:    CFA pattern for teacher's VIF
        weights:    QualityLossWeights instance
        lambda_uv:  [deprecated] if set, overrides weights.w_uv (for compat
                    with call sites still passing lambda_uv=...).
        sample_weights: optional [B] scene/sample weights. Used for all
                    per-sample terms and normalized inside the batch.

    Returns:
        dict with 'loss' (scalar tensor, with grad) and per-term monitoring
        scalars (some with grad, some detached — see keys below).
    """
    w = weights if weights is not None else QualityLossWeights()
    if lambda_uv is not None:
        w = QualityLossWeights(
            w_ssim=w.w_ssim,
            w_vif=w.w_vif,
            w_unique=w.w_unique,
            w_l1_y=w.w_l1_y,
            w_uv=float(lambda_uv),
        )

    from isp.training.training_utils import _ensure_raw_12bit

    raw_12bit = _ensure_raw_12bit(raw_batch)

    device = y_pred.device

    loss_dtype = y_pred.dtype

    ms_ssim_metric = _get_pyiqa_metric("ms_ssim", device)
    ms_ssim_per = ms_ssim_metric(y_pred, y_ref).to(loss_dtype)
    ms_ssim_val = _weighted_mean(ms_ssim_per, sample_weights)

    unique_metric = _get_pyiqa_metric("unique", device)
    rgb_pred = yuv420_to_rgb_bt709_full(y_pred, uv_pred)
    unique_per = unique_metric(rgb_pred).to(loss_dtype)
    unique_val = _weighted_mean(unique_per, sample_weights)

    vif_per = compute_vif_from_raw_and_y_diff(raw_12bit, y_pred, pattern)
    vif_val = _weighted_mean(vif_per.to(loss_dtype), sample_weights)

    l1_uv_per = F.l1_loss(uv_pred, uv_ref, reduction="none")
    l1_uv = _weighted_mean(l1_uv_per, sample_weights)
    l1_y_per = F.l1_loss(y_pred, y_ref, reduction="none")
    l1_y = _weighted_mean(l1_y_per, sample_weights)

    loss = (
        -float(w.w_ssim) * ms_ssim_val
        - float(w.w_vif) * vif_val
        - float(w.w_unique) * unique_val
        + float(w.w_l1_y) * l1_y
        + float(w.w_uv) * l1_uv
    )

    return {
        "loss": loss,
        "ms_ssim": ms_ssim_val.detach(),
        "unique": unique_val.detach(),
        "vif": vif_val.detach(),
        "l1_y": l1_y.detach(),
        "l1_uv": l1_uv.detach(),
    }
