import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sharpening(nn.Module):
    """
    Unsharp mask sharpening with a noise threshold.
    """

    def __init__(self, amount: float = 0.8, radius: float = 1.0, threshold: float = 0.01):
        """
        Args:
            amount: Detail gain; 0 disables sharpening (trainable)
            radius: Gaussian blur sigma in pixels (structural)
            threshold: Minimum |detail| required for sharpening (trainable)
        """
        super().__init__()

        self.amount = nn.Parameter(torch.tensor(float(amount), dtype=torch.float32))
        self.threshold = nn.Parameter(torch.tensor(float(threshold), dtype=torch.float32))

        kernel, kernel_h, kernel_v = self._make_gaussian_kernels(radius)
        self.register_buffer("kernel", kernel)
        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_v", kernel_v)
        self.register_buffer("kernel_h_dw", kernel_h.expand(3, 1, -1, -1).contiguous())
        self.register_buffer("kernel_v_dw", kernel_v.expand(3, 1, -1, -1).contiguous())
        self.pad = kernel.shape[-1] // 2
        self._identity_fast_path = False
        self._refresh_inference_fast_flags()

    def _refresh_inference_fast_flags(self) -> None:
        self._identity_fast_path = bool(self.amount.detach().item() == 0.0)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self._refresh_inference_fast_flags()
        return self

    @staticmethod
    def _make_gaussian_kernels(sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radius = math.ceil(3 * sigma)
        size = 2 * radius + 1
        x = torch.arange(size, dtype=torch.float32) - radius
        g1d = torch.exp(-(x**2) / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        kernel_2d = g2d.unsqueeze(0).unsqueeze(0)
        kernel_h = g1d.view(1, 1, 1, size)
        kernel_v = g1d.view(1, 1, size, 1)
        return kernel_2d, kernel_h, kernel_v

    @staticmethod
    def _make_gaussian_kernel(sigma: float) -> torch.Tensor:
        kernel_2d, _, _ = Sharpening._make_gaussian_kernels(sigma)
        return kernel_2d

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3]
        Returns:
            torch.Tensor: Blurred RGB image [H, W, 3]
        """
        x_4d = x.permute(2, 0, 1).unsqueeze(0)
        x_padded_h = F.pad(x_4d, (self.pad, self.pad, 0, 0), mode="reflect")
        blurred_h = F.conv2d(x_padded_h, self.kernel_h_dw, groups=3)
        x_padded_v = F.pad(blurred_h, (0, 0, self.pad, self.pad), mode="reflect")
        blurred = F.conv2d(x_padded_v, self.kernel_v_dw, groups=3)
        return blurred.squeeze(0).permute(1, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Sharpened RGB image, shape [H, W, 3], float32 in [0, 1]
        """
        if (not self.training) and (not torch.is_grad_enabled()) and self._identity_fast_path:
            return x

        blurred = self._gaussian_blur(x)
        detail = x - blurred

        thresh_safe = self.threshold.clamp(min=0.0)
        abs_d = detail.abs()
        gated = (abs_d - thresh_safe).clamp(min=0.0)
        mask = gated / (gated + thresh_safe + 1e-8)

        output = x + self.amount * detail * mask
        return output.clamp(0.0, 1.0)
