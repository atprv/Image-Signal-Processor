import torch
import torch.nn as nn


class CCM(nn.Module):
    """
    Color correction matrix stage.
    """

    def __init__(self, ccm_config: dict):
        """
        Args:
            ccm_config: CCM configuration dictionary.
        """
        super().__init__()

        ccm_matrix = ccm_config["ccm_matrix"]

        ccm_transposed = ccm_matrix.T

        self.register_buffer("ccm", ccm_transposed)

        self.register_buffer("max_val", torch.tensor(0xFFFFFF, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the CCM to an RGB image.

        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 0xFFFFFF]

        Returns:
            torch.Tensor: CCM-corrected RGB image, shape [H, W, 3], float32 in [0, 1]
        """
        H, W, _ = x.shape

        x_norm = x / self.max_val

        x_flat = x_norm.reshape(-1, 3)
        x_ccm = x_flat @ self.ccm

        x_out = x_ccm.reshape(H, W, 3)

        return torch.clamp(x_out, 0.0, 1.0)
