import torch
import torch.nn as nn


class GammaCorrection(nn.Module):
    """
    Gamma correction for RGB images.
    """

    def __init__(self, gamma: float = 2.2):
        """
        Args:
            gamma: Gamma value
        """
        super().__init__()

        inv_gamma = 1.0 / gamma
        self.register_buffer("inv_gamma", torch.tensor(inv_gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gamma correction.

        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 1]

        Returns:
            torch.Tensor: Gamma-corrected image, shape [H, W, 3], float32 in [0, 1]
        """
        output = x.pow(self.inv_gamma)

        return torch.clamp(output, 0.0, 1.0)
