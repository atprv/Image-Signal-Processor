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

    @staticmethod
    def _validate_params(params: dict):
        """
        Validate ISP parameters before applying them.
        """
        awb_max_gain = params.get("awb_max_gain")
        if awb_max_gain is not None and awb_max_gain < 1.0:
            raise ValueError(f"awb_max_gain must be >= 1.0, got {awb_max_gain}")

        awb_low = params.get("awb_lum_mask_low")
        awb_high = params.get("awb_lum_mask_high")
        if awb_low is not None and not 0.0 <= awb_low <= 1.0:
            raise ValueError(f"awb_lum_mask_low must be in [0, 1], got {awb_low}")
        if awb_high is not None and not 0.0 <= awb_high <= 1.0:
            raise ValueError(f"awb_lum_mask_high must be in [0, 1], got {awb_high}")
        if awb_low is not None and awb_high is not None and awb_low > awb_high:
            raise ValueError(
                f"awb_lum_mask_low must be <= awb_lum_mask_high, got {awb_low} > {awb_high}"
            )

        for key in ["denoise_radius", "ltm_radius", "raw_y_blur_radius", "post_denoise_radius"]:
            value = params.get(key)
            if value is not None and int(value) < 0:
                raise ValueError(f"{key} must be >= 0, got {value}")

        for key in [
            "denoise_eps",
            "sharp_amount",
            "sharp_threshold",
            "ltm_detail_gain",
            "ltm_detail_threshold",
            "post_denoise_eps",
            "saturation",
        ]:
            value = params.get(key)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{key} must be >= 0, got {value}")

        ltm_a = params.get("ltm_a")
        if ltm_a is not None and float(ltm_a) <= 0.0:
            raise ValueError(f"ltm_a must be > 0, got {ltm_a}")

        ltm_eps = params.get("ltm_eps")
        if ltm_eps is not None and float(ltm_eps) <= 0.0:
            raise ValueError(f"ltm_eps must be > 0, got {ltm_eps}")

        ltm_downsample = params.get("ltm_downsample")
        if ltm_downsample is not None and not 0.0 < float(ltm_downsample) <= 1.0:
            raise ValueError(f"ltm_downsample must be in (0, 1], got {ltm_downsample}")

        gamma = params.get("gamma")
        if gamma is not None and float(gamma) <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        sharp_radius = params.get("sharp_radius")
        if sharp_radius is not None and float(sharp_radius) <= 0.0:
            raise ValueError(f"sharp_radius must be > 0, got {sharp_radius}")

        for key in [
            "ltm_target_mean",
            "hist_target_mean",
            "hist_target_std",
            "raw_y_blend",
            "raw_y_full_blend",
        ]:
            value = params.get(key)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1], got {value}")

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
                - raw_y_blend: RAW detail blend into Y
                - raw_y_blur_radius: Blur radius for RAW Y blending
                - saturation: Saturation scaling factor
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
            "raw_y_blend": 0.0,
            "raw_y_blur_radius": 8,
            "saturation": 1.0,
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
        x = self.rgb2yuv(rgb, raw_green=raw_green, full_blend=self.params["raw_y_full_blend"])

        return x

    def _run_to_rgb(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Shared RAW -> RGB float path used by forward() and training helpers.
        """
        x = x.to(self.device)

        # Keep the pre-clamp path for optimization so gradients can flow
        # before display-range clipping.
        x = self.decompand_blacklevel.forward_unclamped(x)
        x = self.denoise(x)  # [H, W] float32 -> [H, W] float32
        x = self.awb(x)  # [H, W] float32 -> [H, W] float32

        raw_green = None
        if self.params["raw_y_blend"] > 0 or self.params["raw_y_full_blend"] > 0:
            raw_green = self.raw_green_extract(x)  # [H, W] float32

        x = self.demosaic(x)  # [H, W] float32 -> [H, W, 3] float32
        x = self.ccm(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.ltm(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]

        x = self.gamma(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.hist_norm(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.post_denoise(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]
        x = self.saturation_adjust(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]

        if raw_green is not None:
            raw_green = raw_green.clamp(0.0, 1.0).pow(self.gamma.inv_gamma)

        if self.params["sharp_amount"] > 0:
            x = self.sharpening(x)  # [H, W, 3] float32 -> [H, W, 3] float32 [0,1]

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
        components = self.rgb2yuv.forward_components(
            rgb, raw_green=raw_green, full_blend=self.params["raw_y_full_blend"]
        )
        nv12 = self.rgb2yuv(rgb, raw_green=raw_green, full_blend=self.params["raw_y_full_blend"])
        return {"y": components["y"], "uv": components["uv"], "nv12": nv12}

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
            "saturation": self.saturation_adjust.saturation.detach().cpu().item(),
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
            "post_denoise_eps": self.post_denoise.eps,
        }

    def set_params(self, **kwargs):
        """
        Set ISP parameters from plain values.
        """
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
            self.awb.max_gain.fill_(kwargs["awb_max_gain"])
        if "awb_lum_mask_low" in kwargs:
            self.awb.lum_mask_low = kwargs["awb_lum_mask_low"]
        if "awb_lum_mask_high" in kwargs:
            self.awb.lum_mask_high = kwargs["awb_lum_mask_high"]

        if "denoise_eps" in kwargs:
            self.denoise.eps.fill_(kwargs["denoise_eps"])
        if "denoise_radius" in kwargs:
            r = int(kwargs["denoise_radius"])
            self.denoise.radius = r
            self.denoise.pad = r
            ks = 2 * r + 1
            kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=self.device) / (ks**2)
            self.denoise.register_buffer("box_kernel", kernel)

        if "ltm_a" in kwargs:
            self.ltm.a.fill_(kwargs["ltm_a"])
        if "ltm_b" in kwargs:
            self.ltm.b.fill_(kwargs["ltm_b"])
        if "ltm_eps" in kwargs:
            self.ltm.eps.fill_(kwargs["ltm_eps"])
        if "ltm_target_mean" in kwargs:
            self.ltm.target_mean.fill_(kwargs["ltm_target_mean"])
        if "ltm_detail_gain" in kwargs:
            self.ltm.detail_gain.fill_(kwargs["ltm_detail_gain"])
        if "ltm_detail_threshold" in kwargs:
            self.ltm.detail_threshold.fill_(kwargs["ltm_detail_threshold"])
        if "ltm_downsample" in kwargs:
            self.ltm.downsample_factor = kwargs["ltm_downsample"]
        if "ltm_radius" in kwargs:
            self.ltm.radius = int(kwargs["ltm_radius"])
        if "ltm_radius" in kwargs or "ltm_downsample" in kwargs:
            self.ltm._rebuild_box_filters()

        if "hist_target_mean" in kwargs:
            self.hist_norm.target_mean = kwargs["hist_target_mean"]
        if "hist_target_std" in kwargs:
            self.hist_norm.target_std = kwargs["hist_target_std"]

        if "sharp_amount" in kwargs:
            self.sharpening.amount.fill_(kwargs["sharp_amount"])
        if "sharp_threshold" in kwargs:
            self.sharpening.threshold.fill_(kwargs["sharp_threshold"])
        if "sharp_radius" in kwargs:
            kernel = Sharpening._make_gaussian_kernel(kwargs["sharp_radius"]).to(self.device)
            self.sharpening.register_buffer("kernel", kernel)
            self.sharpening.pad = kernel.shape[-1] // 2

        if "raw_y_blend" in kwargs:
            self.rgb2yuv.raw_y_blend = kwargs["raw_y_blend"]
        if "raw_y_blur_radius" in kwargs:
            self.rgb2yuv.raw_y_blur_radius = int(kwargs["raw_y_blur_radius"])
        if self.rgb2yuv.raw_y_blend > 0 and (
            "raw_y_blur_radius" in kwargs or "raw_y_blend" in kwargs
        ):
            r = self.rgb2yuv.raw_y_blur_radius
            ks = 2 * r + 1
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32, device=self.device) / ks
            self.rgb2yuv.register_buffer("box_h", box_1d)
            self.rgb2yuv.register_buffer("box_v", box_1d.transpose(2, 3))

        if "post_denoise_eps" in kwargs:
            self.post_denoise.eps = kwargs["post_denoise_eps"]
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
