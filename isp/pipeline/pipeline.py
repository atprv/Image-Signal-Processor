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

        defaults = {
            "awb_max_gain": 4.0,
            "awb_lum_mask_low": 0.0,
            "awb_lum_mask_high": 1.0,
            "denoise_radius": 2,
            "denoise_eps": 100.0,
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

        self.decompand_blacklevel = DecompandBlackLevel(config["decompanding"])

        self.denoise = BayerDenoise(
            radius=self.params["denoise_radius"], eps=self.params["denoise_eps"]
        )

        self.awb = AWB(
            max_gain=self.params["awb_max_gain"],
            lum_mask_low=self.params["awb_lum_mask_low"],
            lum_mask_high=self.params["awb_lum_mask_high"],
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
        x = x.to(self.device)

        x = self.decompand_blacklevel(x)  # [H, W] uint16 -> [H, W] float32
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
        x = self.rgb2yuv(
            x, raw_green=raw_green, full_blend=self.params["raw_y_full_blend"]
        )  # [H, W, 3] float32 -> [N] uint8 (NV12)

        return x

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
