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
            amount: Detail gain; 0 disables sharpening
            radius: Gaussian blur sigma in pixels
            threshold: Minimum |detail| required for sharpening
        """
        super().__init__()

        self.register_buffer("amount", torch.tensor(amount, dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

        kernel = self._make_gaussian_kernel(radius)
        self.register_buffer("kernel", kernel)
        self.pad = kernel.shape[-1] // 2

    @staticmethod
    def _make_gaussian_kernel(sigma: float) -> torch.Tensor:
        radius = math.ceil(3 * sigma)
        size = 2 * radius + 1
        x = torch.arange(size, dtype=torch.float32) - radius
        g1d = torch.exp(-(x**2) / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        return g2d.unsqueeze(0).unsqueeze(0)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3]
        Returns:
            torch.Tensor: Blurred RGB image [H, W, 3]
        """
        x_4d = x.permute(2, 0, 1).unsqueeze(0)
        kernel_dw = self.kernel.expand(3, 1, -1, -1)
        x_padded = F.pad(x_4d, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        blurred = F.conv2d(x_padded, kernel_dw, groups=3)
        return blurred.squeeze(0).permute(1, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Sharpened RGB image, shape [H, W, 3], float32 in [0, 1]
        """
        blurred = self._gaussian_blur(x)
        detail = x - blurred

        mask = (detail.abs() - self.threshold).clamp(min=0.0)
        mask = mask / (mask + self.threshold + 1e-8)

        output = x + self.amount * detail * mask
        return output.clamp(0.0, 1.0)
