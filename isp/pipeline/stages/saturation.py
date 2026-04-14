import torch
import torch.nn as nn


class SaturationAdjust(nn.Module):
    """
    Adjust color saturation in the perceptual domain.
    """

    def __init__(self, saturation: float = 1.0):
        """
        Args:
            saturation: Saturation factor. 1.0 keeps the image unchanged
        """
        super().__init__()
        self.saturation = nn.Parameter(torch.tensor(float(saturation), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Saturation-adjusted RGB [H, W, 3], float32 in [0, 1]
        """
        Y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        Y = Y.unsqueeze(-1)
        x = Y + self.saturation * (x - Y)
        return x.clamp(0.0, 1.0)
