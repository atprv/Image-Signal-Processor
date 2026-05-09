import math

import torch
import torch.nn as nn

from .stages.awb import AWB
from .stages.ccm import CCM
from .stages.decompand import DecompandBlackLevel
from .stages.demosaic import Demosaic
from .stages.denoise import BayerDenoise
from .stages.gamma import GammaCorrection
from .stages.histogram_normalization import HistogramNormalization
from .stages.ltm import LTM
from .stages.post_gamma_denoise import PostGammaDenoise
from .stages.raw_green_extract import RawGreenExtract
from .stages.rgb2yuv import RGBtoYUV
from .stages.saturation import SaturationAdjust
from .stages.sharpening import Sharpening


class ISPPipeline(nn.Module):
    """
    Full ISP pipeline for RAW image processing.
    """

    _VALIDATION_TOL: float = 1e-3
    _POS_FLOOR: float = 1e-6
    _TRAINABLE_PARAM_FALLBACKS: dict[str, float] = {
        "gamma": 2.2,
        "awb_max_gain": 4.0,
        "awb_lum_mask_low": 0.0,
        "awb_lum_mask_high": 1.0,
        "denoise_eps": 1e-6,
        "ltm_a": 0.7,
        "ltm_b": 0.0,
        "ltm_eps": 1e-3,
        "ltm_target_mean": 0.0,
        "ltm_detail_gain": 1.0,
        "ltm_detail_threshold": 0.0,
        "hist_target_mean": 0.0,
        "hist_target_std": 0.0,
        "sharp_amount": 0.0,
        "sharp_threshold": 0.01,
        "saturation": 1.0,
        "raw_y_blend": 0.0,
        "raw_y_full_blend": 0.0,
        "post_denoise_eps": 0.005,
    }

    @staticmethod
    def _validate_params(params: dict):
        """Validate ISP parameters before applying them."""
        tol = ISPPipeline._VALIDATION_TOL

        awb_max_gain = params.get("awb_max_gain")
        if awb_max_gain is not None and float(awb_max_gain) < 1.0 - tol:
            raise ValueError(f"awb_max_gain must be >= 1.0, got {awb_max_gain}")

        awb_low = params.get("awb_lum_mask_low")
        awb_high = params.get("awb_lum_mask_high")
        if awb_low is not None and not -tol <= float(awb_low) <= 1.0 + tol:
            raise ValueError(f"awb_lum_mask_low must be in [0, 1], got {awb_low}")
        if awb_high is not None and not -tol <= float(awb_high) <= 1.0 + tol:
            raise ValueError(f"awb_lum_mask_high must be in [0, 1], got {awb_high}")
        if awb_low is not None and awb_high is not None and float(awb_low) > float(awb_high) + tol:
            raise ValueError(
                f"awb_lum_mask_low must be <= awb_lum_mask_high, got {awb_low} > {awb_high}"
            )

        for key in [
            "denoise_eps",
            "sharp_amount",
            "sharp_threshold",
            "saturation",
            "ltm_detail_gain",
            "ltm_detail_threshold",
            "post_denoise_eps",
        ]:
            value = params.get(key)
            if value is not None and float(value) < -tol:
                raise ValueError(f"{key} must be >= 0, got {value}")

        ltm_a = params.get("ltm_a")
        if ltm_a is not None and float(ltm_a) <= -tol:
            raise ValueError(f"ltm_a must be > 0, got {ltm_a}")

        for key in [
            "ltm_target_mean",
            "hist_target_mean",
            "hist_target_std",
            "raw_y_blend",
            "raw_y_full_blend",
        ]:
            value = params.get(key)
            if value is not None and not -tol <= float(value) <= 1.0 + tol:
                raise ValueError(f"{key} must be in [0, 1], got {value}")

        for key in ["denoise_radius", "ltm_radius", "raw_y_blur_radius", "post_denoise_radius"]:
            value = params.get(key)
            if value is not None and int(value) < 0:
                raise ValueError(f"{key} must be >= 0, got {value}")

        ltm_downsample = params.get("ltm_downsample")
        if ltm_downsample is not None and not 0.0 < float(ltm_downsample) <= 1.0:
            raise ValueError(f"ltm_downsample must be in (0, 1], got {ltm_downsample}")

        sharp_radius = params.get("sharp_radius")
        if sharp_radius is not None and float(sharp_radius) <= 0.0:
            raise ValueError(f"sharp_radius must be > 0, got {sharp_radius}")

        gamma = params.get("gamma")
        if gamma is not None and float(gamma) <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        ltm_eps = params.get("ltm_eps")
        if ltm_eps is not None and float(ltm_eps) <= 0.0:
            raise ValueError(f"ltm_eps must be > 0, got {ltm_eps}")

    @classmethod
    def sanitize_trainable_params_dict(cls, params: dict) -> dict:
        """Clamp learned trainable overrides into the mathematically valid domain."""
        sanitized = dict(params)

        def _read_scalar(key: str):
            if key not in sanitized or sanitized[key] is None:
                return None
            fallback = float(cls._TRAINABLE_PARAM_FALLBACKS[key])
            try:
                value = float(sanitized[key])
            except (TypeError, ValueError):
                value = fallback
            if not math.isfinite(value):
                value = fallback
            return value

        gamma = _read_scalar("gamma")
        if gamma is not None:
            sanitized["gamma"] = max(cls._POS_FLOOR, gamma)

        awb_max_gain = _read_scalar("awb_max_gain")
        if awb_max_gain is not None:
            sanitized["awb_max_gain"] = max(1.0, awb_max_gain)

        awb_low = _read_scalar("awb_lum_mask_low")
        awb_high = _read_scalar("awb_lum_mask_high")
        if awb_low is not None:
            awb_low = min(max(awb_low, 0.0), 1.0)
            sanitized["awb_lum_mask_low"] = awb_low
        if awb_high is not None:
            awb_high = min(max(awb_high, 0.0), 1.0)
            sanitized["awb_lum_mask_high"] = awb_high
        if awb_low is not None and awb_high is not None:
            sanitized["awb_lum_mask_low"] = min(awb_low, awb_high)
            sanitized["awb_lum_mask_high"] = max(awb_low, awb_high)

        denoise_eps = _read_scalar("denoise_eps")
        if denoise_eps is not None:
            sanitized["denoise_eps"] = max(0.0, denoise_eps)

        ltm_a = _read_scalar("ltm_a")
        if ltm_a is not None:
            sanitized["ltm_a"] = max(cls._POS_FLOOR, ltm_a)

        ltm_b = _read_scalar("ltm_b")
        if ltm_b is not None:
            sanitized["ltm_b"] = ltm_b

        ltm_eps = _read_scalar("ltm_eps")
        if ltm_eps is not None:
            sanitized["ltm_eps"] = max(cls._POS_FLOOR, ltm_eps)

        for key in (
            "ltm_target_mean",
            "hist_target_mean",
            "hist_target_std",
            "raw_y_blend",
            "raw_y_full_blend",
        ):
            value = _read_scalar(key)
            if value is not None:
                sanitized[key] = min(max(value, 0.0), 1.0)

        for key in (
            "ltm_detail_gain",
            "ltm_detail_threshold",
            "sharp_amount",
            "sharp_threshold",
            "saturation",
            "post_denoise_eps",
        ):
            value = _read_scalar(key)
            if value is not None:
                sanitized[key] = max(0.0, value)

        return sanitized

    def project_trainable_params_(self) -> None:
        """In-place projection of trainable ISP scalars back into valid ranges."""
        with torch.no_grad():

            def _project_scalar_(
                param: torch.Tensor,
                *,
                fallback: float,
                min_value: float | None = None,
                max_value: float | None = None,
            ) -> None:
                cleaned = torch.nan_to_num(
                    param,
                    nan=float(fallback),
                    posinf=float(max_value if max_value is not None else fallback),
                    neginf=float(min_value if min_value is not None else fallback),
                )
                param.copy_(cleaned)
                clamp_kwargs = {}
                if min_value is not None:
                    clamp_kwargs["min"] = float(min_value)
                if max_value is not None:
                    clamp_kwargs["max"] = float(max_value)
                if clamp_kwargs:
                    param.clamp_(**clamp_kwargs)

            _project_scalar_(
                self.gamma.inv_gamma,
                fallback=1.0 / float(self.params["gamma"]),
                min_value=self._POS_FLOOR,
            )

            _project_scalar_(
                self.awb.max_gain,
                fallback=float(self.params["awb_max_gain"]),
                min_value=1.0,
            )
            _project_scalar_(
                self.awb.lum_mask_low,
                fallback=float(self.params["awb_lum_mask_low"]),
                min_value=0.0,
                max_value=1.0,
            )
            _project_scalar_(
                self.awb.lum_mask_high,
                fallback=float(self.params["awb_lum_mask_high"]),
                min_value=0.0,
                max_value=1.0,
            )
            low = torch.minimum(self.awb.lum_mask_low, self.awb.lum_mask_high)
            high = torch.maximum(self.awb.lum_mask_low, self.awb.lum_mask_high)
            self.awb.lum_mask_low.copy_(low)
            self.awb.lum_mask_high.copy_(high)

            _project_scalar_(
                self.ltm.a,
                fallback=float(self.params["ltm_a"]),
                min_value=self._POS_FLOOR,
            )
            _project_scalar_(
                self.ltm.target_mean,
                fallback=float(self.params["ltm_target_mean"]),
                min_value=0.0,
                max_value=1.0,
            )
            _project_scalar_(
                self.ltm.detail_gain,
                fallback=float(self.params["ltm_detail_gain"]),
                min_value=0.0,
            )
            _project_scalar_(
                self.ltm.detail_threshold,
                fallback=float(self.params["ltm_detail_threshold"]),
                min_value=0.0,
            )

            _project_scalar_(
                self.hist_norm.target_mean,
                fallback=float(self.params["hist_target_mean"]),
                min_value=0.0,
                max_value=1.0,
            )
            _project_scalar_(
                self.hist_norm.target_std,
                fallback=float(self.params["hist_target_std"]),
                min_value=0.0,
                max_value=1.0,
            )

            _project_scalar_(
                self.sharpening.amount,
                fallback=float(self.params["sharp_amount"]),
                min_value=0.0,
            )
            _project_scalar_(
                self.sharpening.threshold,
                fallback=float(self.params["sharp_threshold"]),
                min_value=0.0,
            )
            _project_scalar_(
                self.saturation_adjust.saturation,
                fallback=float(self.params["saturation"]),
                min_value=0.0,
            )

            _project_scalar_(
                self.rgb2yuv.raw_y_blend,
                fallback=float(self.params["raw_y_blend"]),
                min_value=0.0,
                max_value=1.0,
            )
            _project_scalar_(
                self.rgb2yuv.raw_y_full_blend,
                fallback=float(self.params["raw_y_full_blend"]),
                min_value=0.0,
                max_value=1.0,
            )

    def __init__(self, config: dict, device: str = "cuda", **params):
        """
        Initialize the ISP pipeline.

        Args:
            config: Camera configuration dictionary
            device: Compute device ('cuda' or 'cpu')
            **params: ISP stage overrides:
                - awb_max_gain: Maximum AWB gain
                - awb_lum_mask_low: Lower AWB luminance threshold
                - awb_lum_mask_high: Upper AWB luminance threshold
                - denoise_radius: Bayer denoise radius
                - denoise_eps: Bayer denoise regularization
                - ltm_a: LTM compression factor
                - ltm_b: LTM brightness shift
                - ltm_radius: LTM guided-filter radius
                - ltm_downsample: LTM downsample factor
                - ltm_eps: LTM guided-filter epsilon
                - ltm_target_mean: Optional LTM mean target
                - ltm_detail_gain: LTM detail gain
                - ltm_detail_threshold: LTM detail noise threshold
                - hist_target_mean: Histogram mean target
                - hist_target_std: Histogram std target
                - gamma: Gamma value
                - sharp_amount: Sharpening strength
                - sharp_radius: Sharpening blur sigma
                - sharp_threshold: Sharpening detail threshold
                - saturation: Saturation scaling factor
                - raw_y_blend: RAW detail blend into Y
                - raw_y_blur_radius: Blur radius for RAW Y blending
                - raw_y_full_blend: Full RAW blend into Y
                - post_denoise_radius: Post-gamma denoise radius
                - post_denoise_eps: Post-gamma denoise epsilon
        """
        super().__init__()

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)

        self.decompand_blacklevel = DecompandBlackLevel(config["decompanding"])

        defaults = {
            "awb_max_gain": 4.0,
            "awb_lum_mask_low": 0.0,
            "awb_lum_mask_high": 1.0,
            "denoise_radius": 2,
            "denoise_eps": 1e-6,
            "ltm_a": 0.7,
            "ltm_b": 0.0,
            "ltm_radius": 8,
            "ltm_downsample": 0.5,
            "ltm_eps": 1e-3,
            "ltm_target_mean": 0.0,
            "ltm_detail_gain": 1.0,
            "ltm_detail_threshold": 0.0,
            "hist_target_mean": 0.0,
            "hist_target_std": 0.0,
            "gamma": 2.2,
            "sharp_amount": 0.0,
            "sharp_radius": 1.0,
            "sharp_threshold": 0.01,
            "saturation": 1.0,
            "raw_y_blend": 0.0,
            "raw_y_blur_radius": 8,
            "raw_y_full_blend": 0.0,
            "post_denoise_radius": 0,
            "post_denoise_eps": 0.005,
        }

        self.params = {**defaults, **params}
        self._validate_params(self.params)
        awb_eps_norm = 1.0 / float(self.decompand_blacklevel.output_scale.item())

        self.denoise = BayerDenoise(
            radius=self.params["denoise_radius"], eps=self.params["denoise_eps"]
        )

        self.awb = AWB(
            max_gain=self.params["awb_max_gain"],
            lum_mask_low=self.params["awb_lum_mask_low"],
            lum_mask_high=self.params["awb_lum_mask_high"],
            eps=awb_eps_norm,
        )

        self.demosaic = Demosaic()

        self.ccm = CCM(config["ccm"])

        self.ltm = LTM(
            a=self.params["ltm_a"],
            b=self.params["ltm_b"],
            radius=self.params["ltm_radius"],
            eps=self.params["ltm_eps"],
            downsample_factor=self.params["ltm_downsample"],
            target_mean=self.params["ltm_target_mean"],
            detail_gain=self.params["ltm_detail_gain"],
            detail_threshold=self.params["ltm_detail_threshold"],
        )

        self.gamma = GammaCorrection(gamma=self.params["gamma"])

        self.hist_norm = HistogramNormalization(
            target_mean=self.params["hist_target_mean"], target_std=self.params["hist_target_std"]
        )

        self.post_denoise = PostGammaDenoise(
            radius=self.params["post_denoise_radius"], eps=self.params["post_denoise_eps"]
        )

        self.saturation_adjust = SaturationAdjust(saturation=self.params["saturation"])

        self.sharpening = Sharpening(
            amount=self.params["sharp_amount"],
            radius=self.params["sharp_radius"],
            threshold=self.params["sharp_threshold"],
        )

        self.raw_green_extract = RawGreenExtract()

        self.rgb2yuv = RGBtoYUV(
            raw_y_blend=self.params["raw_y_blend"],
            raw_y_blur_radius=self.params["raw_y_blur_radius"],
            raw_y_full_blend=self.params["raw_y_full_blend"],
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one RAW frame through the full ISP pipeline.

        Args:
            x: RAW Bayer RGGB frame, shape [H, W], dtype uint16

        Returns:
            torch.Tensor: Processed NV12 frame as a 1D uint8 tensor
        """
        rgb, raw_green = self._run_to_rgb(x)
        x = self.rgb2yuv(rgb, raw_green=raw_green)

        return x

    def _inference_fast_path_enabled(self) -> bool:
        """Enable inference-only shortcuts that are safe when gradients are off."""
        return (not self.training) and (not torch.is_grad_enabled())

    def _run_to_rgb(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shared RAW -> RGB float path used by forward() and training helpers."""
        x = x.to(self.device)

        x = self.decompand_blacklevel.forward_unclamped(x)
        x = self.denoise(x)
        x = self.awb(x)

        need_raw_green = True
        if self._inference_fast_path_enabled() and self.rgb2yuv.raw_green_blend_is_identity():
            need_raw_green = False

        raw_green = self.raw_green_extract(x) if need_raw_green else None

        x = self.demosaic(x)
        x = self.ccm(x)
        x = self.ltm(x)

        x = self.gamma(x)
        x = self.hist_norm(x)
        x = self.post_denoise(x)
        x = self.saturation_adjust(x)

        if raw_green is not None:
            raw_green = raw_green.clamp(min=1e-6, max=1.0).pow(self.gamma.inv_gamma)

        x = self.sharpening(x)

        return x, raw_green

    def forward_components(self, x: torch.Tensor) -> dict:
        """
        Differentiable forward pass returning float Y/UV and packed NV12.

        Args:
            x: RAW Bayer RGGB frame, shape [H, W]

        Returns:
            dict with:
                y: [1,1,H,W] float32 [0,1]
                uv: [1,2,H/2,W/2] float32 [0,1]
                nv12: packed NV12 uint8 tensor matching forward()
        """
        rgb, raw_green = self._run_to_rgb(x)
        components = self.rgb2yuv.forward_components(rgb, raw_green=raw_green)
        return components

    def forward_diff(self, x: torch.Tensor) -> dict:
        """
        Differentiable forward pass returning float Y and UV.

        Args:
            x: RAW Bayer RGGB frame, shape [H, W]

        Returns:
            dict with y: [1,1,H,W] float32 [0,1], uv: [1,2,H/2,W/2] float32 [0,1]
        """
        components = self.forward_components(x)
        return {"y": components["y"], "uv": components["uv"]}

    def get_params(self) -> dict:
        """
        Return all current ISP parameters as plain Python values.
        """
        return {
            "ccm_matrix": self.ccm.ccm.detach().cpu().T.tolist(),
            "gamma": 1.0 / self.gamma.inv_gamma.detach().cpu().item(),
            "saturation": float(self.saturation_adjust.saturation.item()),
            "awb_max_gain": float(self.awb.max_gain.item()),
            "awb_lum_mask_low": self.awb.lum_mask_low,
            "awb_lum_mask_high": self.awb.lum_mask_high,
            "denoise_radius": self.denoise.radius,
            "denoise_eps": float(self.denoise.eps.item()),
            "ltm_a": float(self.ltm.a.item()),
            "ltm_b": float(self.ltm.b.item()),
            "ltm_radius": self.ltm.radius,
            "ltm_downsample": self.ltm.downsample_factor,
            "ltm_eps": float(self.ltm.eps.item()),
            "ltm_target_mean": float(self.ltm.target_mean.item()),
            "ltm_detail_gain": float(self.ltm.detail_gain.item()),
            "ltm_detail_threshold": float(self.ltm.detail_threshold.item()),
            "hist_target_mean": self.hist_norm.target_mean,
            "hist_target_std": self.hist_norm.target_std,
            "sharp_amount": float(self.sharpening.amount.item()),
            "sharp_radius": self.params["sharp_radius"],
            "sharp_threshold": float(self.sharpening.threshold.item()),
            "raw_y_blend": self.rgb2yuv.raw_y_blend,
            "raw_y_blur_radius": self.rgb2yuv.raw_y_blur_radius,
            "raw_y_full_blend": self.params["raw_y_full_blend"],
            "post_denoise_radius": self.post_denoise.radius,
            "post_denoise_eps": float(self.post_denoise.eps.item()),
        }

    def set_params(self, **kwargs):
        """Set ISP parameters from plain values."""
        updated_params = {**self.params, **kwargs}
        self._validate_params(updated_params)
        self.params = updated_params

        if "ccm_matrix" in kwargs:
            mat = torch.tensor(
                kwargs["ccm_matrix"], dtype=torch.float32, device=self.ccm.ccm.device
            ).T
            self.ccm.ccm.data.copy_(mat)
        if "gamma" in kwargs:
            self.gamma.inv_gamma.data.fill_(1.0 / kwargs["gamma"])
        if "saturation" in kwargs:
            self.saturation_adjust.saturation.data.fill_(kwargs["saturation"])

        if "awb_max_gain" in kwargs:
            self.awb.max_gain.data.fill_(kwargs["awb_max_gain"])
        if "awb_lum_mask_low" in kwargs:
            self.awb.lum_mask_low.data.fill_(kwargs["awb_lum_mask_low"])
        if "awb_lum_mask_high" in kwargs:
            self.awb.lum_mask_high.data.fill_(kwargs["awb_lum_mask_high"])

        if "denoise_eps" in kwargs:
            self.denoise.set_eps(kwargs["denoise_eps"])
        if "denoise_radius" in kwargs:
            r = int(kwargs["denoise_radius"])
            self.denoise.radius = r
            self.denoise.pad = r
            ks = 2 * r + 1
            kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=self.device) / (ks**2)
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32, device=self.device) / ks
            self.denoise.register_buffer("box_kernel", kernel)
            self.denoise.register_buffer("box_h", box_1d)
            self.denoise.register_buffer("box_v", box_1d.transpose(2, 3))

        if "ltm_a" in kwargs:
            self.ltm.a.data.fill_(kwargs["ltm_a"])
        if "ltm_b" in kwargs:
            self.ltm.b.data.fill_(kwargs["ltm_b"])
        if "ltm_eps" in kwargs:
            self.ltm.set_eps(kwargs["ltm_eps"])
        if "ltm_target_mean" in kwargs:
            self.ltm.target_mean.data.fill_(kwargs["ltm_target_mean"])
        if "ltm_detail_gain" in kwargs:
            self.ltm.detail_gain.data.fill_(kwargs["ltm_detail_gain"])
        if "ltm_detail_threshold" in kwargs:
            self.ltm.detail_threshold.data.fill_(kwargs["ltm_detail_threshold"])
        if "ltm_downsample" in kwargs:
            self.ltm.downsample_factor = kwargs["ltm_downsample"]
        if "ltm_radius" in kwargs:
            self.ltm.radius = int(kwargs["ltm_radius"])
        if "ltm_radius" in kwargs or "ltm_downsample" in kwargs:
            self.ltm._rebuild_box_filters()

        if "hist_target_mean" in kwargs:
            self.hist_norm.target_mean.data.fill_(kwargs["hist_target_mean"])
        if "hist_target_std" in kwargs:
            self.hist_norm.target_std.data.fill_(kwargs["hist_target_std"])
        if "hist_target_mean" in kwargs or "hist_target_std" in kwargs:
            refresh_hist = getattr(self.hist_norm, "_refresh_inference_fast_flags", None)
            if callable(refresh_hist):
                refresh_hist()

        if "sharp_amount" in kwargs:
            self.sharpening.amount.data.fill_(kwargs["sharp_amount"])
        if "sharp_threshold" in kwargs:
            self.sharpening.threshold.data.fill_(kwargs["sharp_threshold"])
        if "sharp_radius" in kwargs:
            kernel, kernel_h, kernel_v = Sharpening._make_gaussian_kernels(kwargs["sharp_radius"])
            kernel = kernel.to(self.device)
            kernel_h = kernel_h.to(self.device)
            kernel_v = kernel_v.to(self.device)
            kernel_h_dw = kernel_h.expand(3, 1, -1, -1).contiguous()
            kernel_v_dw = kernel_v.expand(3, 1, -1, -1).contiguous()
            self.sharpening.register_buffer("kernel", kernel)
            self.sharpening.register_buffer("kernel_h", kernel_h)
            self.sharpening.register_buffer("kernel_v", kernel_v)
            self.sharpening.register_buffer("kernel_h_dw", kernel_h_dw)
            self.sharpening.register_buffer("kernel_v_dw", kernel_v_dw)
            self.sharpening.pad = kernel.shape[-1] // 2
        if "sharp_amount" in kwargs:
            refresh_sharp = getattr(self.sharpening, "_refresh_inference_fast_flags", None)
            if callable(refresh_sharp):
                refresh_sharp()

        if "raw_y_blend" in kwargs:
            self.rgb2yuv.raw_y_blend.data.fill_(kwargs["raw_y_blend"])
        if "raw_y_full_blend" in kwargs:
            self.rgb2yuv.raw_y_full_blend.data.fill_(kwargs["raw_y_full_blend"])
        if "raw_y_blur_radius" in kwargs:
            self.rgb2yuv.raw_y_blur_radius = int(kwargs["raw_y_blur_radius"])
            r = self.rgb2yuv.raw_y_blur_radius
            ks = 2 * r + 1
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32, device=self.device) / ks
            self.rgb2yuv.register_buffer("box_h", box_1d)
            self.rgb2yuv.register_buffer("box_v", box_1d.transpose(2, 3))
        if "raw_y_blend" in kwargs or "raw_y_full_blend" in kwargs:
            refresh_rgb2yuv = getattr(self.rgb2yuv, "_refresh_inference_fast_flags", None)
            if callable(refresh_rgb2yuv):
                refresh_rgb2yuv()

        if "post_denoise_eps" in kwargs:
            self.post_denoise.set_eps(kwargs["post_denoise_eps"])
        if "post_denoise_radius" in kwargs:
            r = int(kwargs["post_denoise_radius"])
            self.post_denoise.radius = r
            if r > 0:
                ks = 2 * r + 1
                box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32, device=self.device) / ks
                self.post_denoise.register_buffer("box_h", box_1d)
                self.post_denoise.register_buffer("box_v", box_1d.transpose(2, 3))

    def get_pipeline_info(self) -> dict:
        """
        Return pipeline configuration details.

        Returns:
            dict: Active device, parameters, and stage list.
        """
        return {
            "device": str(self.device),
            "parameters": self.params,
            "modules": [
                "DecompandBlackLevel",
                "BayerDenoise",
                "AWB",
                "RawGreenExtract",
                "Demosaic",
                "CCM",
                "LTM",
                "GammaCorrection",
                "HistogramNormalization",
                "PostGammaDenoise",
                "SaturationAdjust",
                "Sharpening",
                "RGBtoYUV",
            ],
        }
