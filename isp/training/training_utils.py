from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Adam

from isp.color.conversions import yuv420_to_yuv444, yuv444_to_yuv420
from isp.evaluation.evaluation_utils import (
    compute_vif_from_raw_and_y,
    run_isp_frame,
)


def _infer_module_device(*modules, default: str = "cpu") -> torch.device:
    """
    Infer device from module parameters or buffers.
    """
    for module in modules:
        if module is None:
            continue

        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                return parameter.device

        if hasattr(module, "buffers"):
            for buffer in module.buffers():
                return buffer.device

    return torch.device(default)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    Move tensor values in a batch dict to the target device.
    """
    moved_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def _ensure_raw_12bit(raw_batch: torch.Tensor) -> torch.Tensor:
    """
    Convert RAW batch to 12-bit sensor range [0, 4095].
    """
    if raw_batch.ndim != 4 or raw_batch.shape[1] != 1:
        raise ValueError(
            f"Expected RAW batch with shape [B, 1, H, W], got {tuple(raw_batch.shape)}"
        )

    raw_batch = raw_batch.to(torch.float32)

    if raw_batch.max().item() <= 1.0:
        raw_12bit = (raw_batch * 4095.0).round()
    else:
        raw_12bit = raw_batch

    return raw_12bit.clamp(0.0, 4095.0)


TRAINABLE_ISP_PARAM_KEYS = {"ccm_matrix", "gamma", "saturation"}


def _apply_scene_params_preserving_trainables(isp, scene_params: dict | None):
    """
    Apply scene-specific non-differentiable ISP knobs without overwriting the
    shared trainable CCM/gamma/saturation parameters.
    """
    if not scene_params:
        return
    non_trainable = {
        key: value for key, value in scene_params.items() if key not in TRAINABLE_ISP_PARAM_KEYS
    }
    if non_trainable:
        isp.set_params(**non_trainable)


def _run_isp_batch(isp, raw_12bit_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Run ISP frame-by-frame on a batch of RAW patches.
    """
    if raw_12bit_batch.ndim != 4 or raw_12bit_batch.shape[1] != 1:
        raise ValueError(
            f"Expected RAW batch with shape [B, 1, H, W], got {tuple(raw_12bit_batch.shape)}"
        )

    batch_size, _, height, width = raw_12bit_batch.shape

    y_frames = []
    uv_frames = []

    for batch_index in range(batch_size):
        raw_frame = raw_12bit_batch[batch_index, 0]
        y_isp, uv_isp = run_isp_frame(isp, raw_frame, width=width, height=height)
        y_frames.append(y_isp)
        uv_frames.append(uv_isp)

    y_batch = torch.cat(y_frames, dim=0)
    uv_batch = torch.cat(uv_frames, dim=0)

    return {
        "y_isp": y_batch,
        "uv_isp": uv_batch,
    }


def forward_isp_cnn(isp, cnn, raw_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Run RAW -> ISP -> CNN residual correction for a batch of patches.
    """
    if raw_batch.ndim != 4 or raw_batch.shape[1] != 1:
        raise ValueError(
            f"Expected RAW batch with shape [B, 1, H, W], got {tuple(raw_batch.shape)}"
        )

    device = _infer_module_device(cnn, isp, default=str(raw_batch.device))
    raw_batch = raw_batch.to(device)
    raw_12bit = _ensure_raw_12bit(raw_batch)

    isp_outputs = _run_isp_batch(isp, raw_12bit)
    y_isp = isp_outputs["y_isp"]
    uv_isp = isp_outputs["uv_isp"]

    yuv444_isp = yuv420_to_yuv444(y_isp, uv_isp)
    residual = cnn(yuv444_isp)
    yuv444_pred = torch.clamp(yuv444_isp + residual, 0.0, 1.0)
    y_pred, uv_pred = yuv444_to_yuv420(yuv444_pred)

    return {
        "raw_12bit": raw_12bit,
        "y_isp": y_isp,
        "uv_isp": uv_isp,
        "yuv444_isp": yuv444_isp,
        "residual": residual,
        "yuv444_pred": yuv444_pred,
        "y_pred": y_pred,
        "uv_pred": uv_pred,
    }


def compute_proxy_loss(
    raw_batch: torch.Tensor,
    y_pred: torch.Tensor,
    uv_pred: torch.Tensor,
    y_ref: torch.Tensor,
    uv_ref: torch.Tensor,
    pattern: str,
    lambda_uv: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Compute differentiable proxy loss and VIF monitoring metric.
    """
    raw_12bit = _ensure_raw_12bit(raw_batch)

    l1_y = F.l1_loss(y_pred, y_ref)
    l1_uv = F.l1_loss(uv_pred, uv_ref)
    loss = l1_y + float(lambda_uv) * l1_uv

    with torch.no_grad():
        batch_size = raw_12bit.shape[0]
        vif_sum = 0.0
        for i in range(batch_size):
            vif_val = compute_vif_from_raw_and_y(
                raw_12bit[i : i + 1],
                y_pred[i : i + 1].detach(),
                pattern,
            )
            vif_sum += vif_val.item()
        vif_mean = torch.tensor(vif_sum / batch_size)

    return {
        "loss": loss,
        "l1_y": l1_y,
        "l1_uv": l1_uv,
        "vif": vif_mean,
    }


def _get_grad_mean(parameter: torch.nn.Parameter | None) -> float | None:
    if parameter is None or parameter.grad is None:
        return None
    return float(parameter.grad.abs().mean().item())


def train_step(
    isp, cnn, optimizer, batch: dict[str, Any], pattern: str, lambda_uv: float = 1.0
) -> dict[str, float]:
    """
    Run one optimization step for CNN on top of fixed ISP outputs.
    """
    if optimizer is None:
        raise ValueError("optimizer must not be None")

    device = _infer_module_device(cnn, isp, default="cpu")
    batch = _move_batch_to_device(batch, device)

    raw = batch["raw"]
    y_ref = batch["y_ref"]
    uv_ref = batch["uv_ref"]

    if hasattr(cnn, "train"):
        cnn.train()
    if hasattr(isp, "eval"):
        isp.eval()

    optimizer.zero_grad(set_to_none=True)

    outputs = forward_isp_cnn(isp, cnn, raw)
    loss_dict = compute_proxy_loss(
        raw_batch=outputs["raw_12bit"],
        y_pred=outputs["y_pred"],
        uv_pred=outputs["uv_pred"],
        y_ref=y_ref,
        uv_ref=uv_ref,
        pattern=pattern,
        lambda_uv=lambda_uv,
    )

    loss = loss_dict["loss"]
    loss.backward()

    head_grad = _get_grad_mean(cnn.head[0].weight if hasattr(cnn, "head") else None)
    body_grad = None
    if hasattr(cnn, "body") and len(cnn.body) > 0:
        first_block = cnn.body[0]
        if hasattr(first_block, "conv1"):
            body_grad = _get_grad_mean(first_block.conv1.weight)
    tail_grad = _get_grad_mean(cnn.tail.weight if hasattr(cnn, "tail") else None)

    optimizer.step()

    return {
        "loss": float(loss.item()),
        "l1_y": float(loss_dict["l1_y"].item()),
        "l1_uv": float(loss_dict["l1_uv"].item()),
        "vif": float(loss_dict["vif"].item()),
        "head_grad_mean": head_grad,
        "body_grad_mean": body_grad,
        "tail_grad_mean": tail_grad,
    }


def _run_isp_batch_diff(
    isp,
    raw_12bit_batch: torch.Tensor,
    scene_ids: torch.Tensor | None = None,
    scene_params_by_id: dict[int, dict] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run ISP frame-by-frame using forward_diff so gradients reach ISP parameters.
    """
    if raw_12bit_batch.ndim != 4 or raw_12bit_batch.shape[1] != 1:
        raise ValueError(
            f"Expected RAW batch with shape [B, 1, H, W], got {tuple(raw_12bit_batch.shape)}"
        )

    if not hasattr(isp, "forward_diff"):
        raise AttributeError(
            "ISP must implement forward_diff(raw) returning {'y': float, 'uv': float} "
            "for E2E training. Got module without forward_diff."
        )

    batch_size = raw_12bit_batch.shape[0]
    y_frames = []
    uv_frames = []

    for batch_index in range(batch_size):
        if scene_ids is not None and scene_params_by_id is not None:
            scene_id = int(scene_ids[batch_index].item())
            _apply_scene_params_preserving_trainables(
                isp,
                scene_params_by_id.get(scene_id),
            )
        raw_frame = raw_12bit_batch[batch_index, 0]
        out = isp.forward_diff(raw_frame)
        y_frames.append(out["y"])
        uv_frames.append(out["uv"])

    return {
        "y_isp": torch.cat(y_frames, dim=0),
        "uv_isp": torch.cat(uv_frames, dim=0),
    }


def forward_isp_cnn_diff(
    isp,
    cnn,
    raw_batch: torch.Tensor,
    scene_ids: torch.Tensor | None = None,
    scene_params_by_id: dict[int, dict] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Differentiable RAW -> ISP -> CNN forward path. Gradients flow into both.
    """
    if raw_batch.ndim != 4 or raw_batch.shape[1] != 1:
        raise ValueError(
            f"Expected RAW batch with shape [B, 1, H, W], got {tuple(raw_batch.shape)}"
        )

    device = _infer_module_device(cnn, isp, default=str(raw_batch.device))
    raw_batch = raw_batch.to(device)
    raw_12bit = _ensure_raw_12bit(raw_batch)

    isp_outputs = _run_isp_batch_diff(
        isp,
        raw_12bit,
        scene_ids=scene_ids,
        scene_params_by_id=scene_params_by_id,
    )
    y_isp = isp_outputs["y_isp"]
    uv_isp = isp_outputs["uv_isp"]

    yuv444_isp = yuv420_to_yuv444(y_isp, uv_isp)
    residual = cnn(yuv444_isp)
    yuv444_pred = torch.clamp(yuv444_isp + residual, 0.0, 1.0)
    y_pred, uv_pred = yuv444_to_yuv420(yuv444_pred)

    return {
        "raw_12bit": raw_12bit,
        "y_isp": y_isp,
        "uv_isp": uv_isp,
        "yuv444_isp": yuv444_isp,
        "residual": residual,
        "yuv444_pred": yuv444_pred,
        "y_pred": y_pred,
        "uv_pred": uv_pred,
    }


def compute_isp_anchor_loss(
    isp,
    anchor_params: dict[str, Any],
    gamma_scale: float = 0.1,
    saturation_scale: float = 0.1,
    ccm_scale: float = 0.05,
) -> torch.Tensor:
    """
    Penalize drift of the trainable ISP parameters from their initial values.
    """
    device = isp.ccm.ccm.device
    dtype = isp.ccm.ccm.dtype
    terms = []

    if "ccm_matrix" in anchor_params and hasattr(isp, "ccm"):
        target_ccm = torch.as_tensor(
            anchor_params["ccm_matrix"],
            device=device,
            dtype=dtype,
        )
        current_ccm = isp.ccm.ccm.T
        terms.append(F.mse_loss(current_ccm / ccm_scale, target_ccm / ccm_scale))

    if "gamma" in anchor_params and hasattr(isp, "gamma"):
        target_gamma = torch.as_tensor(
            float(anchor_params["gamma"]),
            device=device,
            dtype=dtype,
        )
        current_gamma = 1.0 / isp.gamma.inv_gamma.clamp_min(1e-6)
        terms.append(F.mse_loss(current_gamma / gamma_scale, target_gamma / gamma_scale))

    if "saturation" in anchor_params and hasattr(isp, "saturation_adjust"):
        target_saturation = torch.as_tensor(
            float(anchor_params["saturation"]),
            device=device,
            dtype=dtype,
        )
        current_saturation = isp.saturation_adjust.saturation
        terms.append(
            F.mse_loss(current_saturation / saturation_scale, target_saturation / saturation_scale)
        )

    if not terms:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.stack([term.reshape(()) for term in terms]).mean()


def e2e_train_step(
    isp,
    cnn,
    optimizer,
    batch: dict[str, Any],
    pattern: str,
    lambda_uv: float = 1.0,
    loss_type: str = "proxy",
    quality_weights: Any | None = None,
    isp_anchor_params: dict[str, Any] | None = None,
    isp_reg_weight: float = 0.0,
    isp_reg_gamma_scale: float = 0.1,
    isp_reg_saturation_scale: float = 0.1,
    isp_reg_ccm_scale: float = 0.05,
    scene_params_by_id: dict[int, dict] | None = None,
    scene_loss_weights_by_id: dict[int, float] | None = None,
) -> dict[str, float | None]:
    """
    Run one E2E optimization step where gradients reach BOTH ISP and CNN.

    Args:
        loss_type: 'proxy'  -> L1(Y) + lambda_uv * L1(UV)
                   'quality' -> -w_ssim*MS_SSIM - w_vif*VIF - w_unique*UNIQUE
                                + w_uv*L1(UV)
        quality_weights: QualityLossWeights; used only when loss_type=='quality'.
                         None defaults per the dataclass.
        lambda_uv: for 'proxy' loss: L1(UV) weight. For 'quality' loss: overrides
                   w_uv in quality_weights (kept for call-site compatibility).
        isp_anchor_params: initial ISP params for soft anti-drift regularization.
        isp_reg_weight: multiplies the anchor regularizer. 0 disables it.
        scene_params_by_id: optional full per-scene ISP knob dicts keyed by
                   scene_id. Trainable params are preserved.
        scene_loss_weights_by_id: optional loss weights keyed by scene_id.

    Returns:
        Dict with unified keys across both loss types. Quality-only metrics
        (ms_ssim, unique) are None when loss_type=='proxy'.
    """
    if optimizer is None:
        raise ValueError("optimizer must not be None")
    if loss_type not in ("proxy", "quality"):
        raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'proxy' or 'quality'")

    device = _infer_module_device(cnn, isp, default="cpu")
    batch = _move_batch_to_device(batch, device)

    raw = batch["raw"]
    y_ref = batch["y_ref"]
    uv_ref = batch["uv_ref"]

    if hasattr(cnn, "train"):
        cnn.train()
    if hasattr(isp, "train"):
        isp.train()

    optimizer.zero_grad(set_to_none=True)

    scene_ids = batch.get("scene_id")
    if torch.is_tensor(scene_ids):
        scene_ids = scene_ids.to(device)
    else:
        scene_ids = None

    sample_weights = None
    if scene_ids is not None and scene_loss_weights_by_id:
        sample_weights = torch.ones(
            scene_ids.shape[0],
            device=device,
            dtype=torch.float32,
        )
        for scene_id, weight in scene_loss_weights_by_id.items():
            sample_weights = torch.where(
                scene_ids == int(scene_id),
                torch.full_like(sample_weights, float(weight)),
                sample_weights,
            )

    def _scalar(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return float(v.item())
        return float(v)

    if loss_type == "quality":
        from isp.training.quality_loss import (
            QualityLossWeights,
            compute_quality_loss,
        )

        w = quality_weights if quality_weights is not None else QualityLossWeights()
    else:
        compute_quality_loss = None
        w = None

    scene_aware = scene_ids is not None and scene_params_by_id is not None
    isp_anchor_reg = None

    if scene_aware:
        total_weight = (
            sample_weights.sum().clamp_min(1e-8)
            if sample_weights is not None
            else torch.tensor(float(raw.shape[0]), device=device)
        )
        accum = {
            "loss": 0.0,
            "l1_y": 0.0,
            "l1_uv": 0.0,
            "vif": 0.0,
            "ms_ssim": 0.0,
            "unique": 0.0,
        }
        for batch_index in range(raw.shape[0]):
            scene_id = int(scene_ids[batch_index].item())
            _apply_scene_params_preserving_trainables(
                isp,
                scene_params_by_id.get(scene_id),
            )
            outputs_i = forward_isp_cnn_diff(isp, cnn, raw[batch_index : batch_index + 1])
            if loss_type == "proxy":
                loss_dict_i = compute_proxy_loss(
                    raw_batch=outputs_i["raw_12bit"],
                    y_pred=outputs_i["y_pred"],
                    uv_pred=outputs_i["uv_pred"],
                    y_ref=y_ref[batch_index : batch_index + 1],
                    uv_ref=uv_ref[batch_index : batch_index + 1],
                    pattern=pattern,
                    lambda_uv=lambda_uv,
                )
            else:
                loss_dict_i = compute_quality_loss(
                    raw_batch=outputs_i["raw_12bit"],
                    y_pred=outputs_i["y_pred"],
                    uv_pred=outputs_i["uv_pred"],
                    y_ref=y_ref[batch_index : batch_index + 1],
                    uv_ref=uv_ref[batch_index : batch_index + 1],
                    pattern=pattern,
                    weights=w,
                    lambda_uv=None,
                    sample_weights=None,
                )
            scale = (
                sample_weights[batch_index] / total_weight
                if sample_weights is not None
                else torch.tensor(1.0 / raw.shape[0], device=device)
            )
            (loss_dict_i["loss"] * scale).backward()
            scale_f = float(scale.detach().item())
            for key in accum:
                value = _scalar(loss_dict_i.get(key))
                if value is not None:
                    accum[key] += scale_f * value
        loss_dict = {key: torch.tensor(value, device=device) for key, value in accum.items()}
        loss = loss_dict["loss"]
    else:
        outputs = forward_isp_cnn_diff(isp, cnn, raw)

        if loss_type == "proxy":
            loss_dict = compute_proxy_loss(
                raw_batch=outputs["raw_12bit"],
                y_pred=outputs["y_pred"],
                uv_pred=outputs["uv_pred"],
                y_ref=y_ref,
                uv_ref=uv_ref,
                pattern=pattern,
                lambda_uv=lambda_uv,
            )
        else:
            loss_dict = compute_quality_loss(
                raw_batch=outputs["raw_12bit"],
                y_pred=outputs["y_pred"],
                uv_pred=outputs["uv_pred"],
                y_ref=y_ref,
                uv_ref=uv_ref,
                pattern=pattern,
                weights=w,
                lambda_uv=None,
                sample_weights=sample_weights,
            )

        loss = loss_dict["loss"]
        loss.backward()

    if isp_anchor_params is not None and float(isp_reg_weight) > 0.0:
        isp_anchor_reg = compute_isp_anchor_loss(
            isp=isp,
            anchor_params=isp_anchor_params,
            gamma_scale=isp_reg_gamma_scale,
            saturation_scale=isp_reg_saturation_scale,
            ccm_scale=isp_reg_ccm_scale,
        )
        (float(isp_reg_weight) * isp_anchor_reg).backward()
        loss = loss + float(isp_reg_weight) * isp_anchor_reg

    head_grad = _get_grad_mean(cnn.head[0].weight if hasattr(cnn, "head") else None)
    body_grad = None
    if hasattr(cnn, "body") and len(cnn.body) > 0:
        first_block = cnn.body[0]
        if hasattr(first_block, "conv1"):
            body_grad = _get_grad_mean(first_block.conv1.weight)
    tail_grad = _get_grad_mean(cnn.tail.weight if hasattr(cnn, "tail") else None)

    isp_ccm_grad = _get_grad_mean(isp.ccm.ccm) if hasattr(isp, "ccm") else None
    isp_gamma_grad = _get_grad_mean(isp.gamma.inv_gamma) if hasattr(isp, "gamma") else None
    isp_saturation_grad = (
        _get_grad_mean(isp.saturation_adjust.saturation)
        if hasattr(isp, "saturation_adjust")
        else None
    )

    optimizer.step()

    return {
        "loss": float(loss.item()),
        "l1_y": _scalar(loss_dict.get("l1_y")),
        "l1_uv": _scalar(loss_dict.get("l1_uv")),
        "vif": _scalar(loss_dict.get("vif")),
        "ms_ssim": _scalar(loss_dict.get("ms_ssim")),
        "unique": _scalar(loss_dict.get("unique")),
        "isp_anchor_reg": _scalar(isp_anchor_reg),
        "head_grad_mean": head_grad,
        "body_grad_mean": body_grad,
        "tail_grad_mean": tail_grad,
        "isp_ccm_grad_mean": isp_ccm_grad,
        "isp_gamma_grad_mean": isp_gamma_grad,
        "isp_saturation_grad_mean": isp_saturation_grad,
    }


def overfit_one_batch(
    isp,
    cnn,
    batch: dict[str, Any],
    pattern: str,
    steps: int = 100,
    lr: float = 1e-3,
    lambda_uv: float = 1.0,
) -> list[dict[str, float]]:
    """
    Overfit CNN on one batch to verify trainability.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    trainable_parameters = [parameter for parameter in cnn.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("cnn has no trainable parameters")

    optimizer = Adam(trainable_parameters, lr=lr)
    history: list[dict[str, float]] = []

    for step_index in range(steps):
        metrics = train_step(
            isp=isp, cnn=cnn, optimizer=optimizer, batch=batch, pattern=pattern, lambda_uv=lambda_uv
        )
        metrics["step"] = step_index + 1
        history.append(metrics)

    return history
