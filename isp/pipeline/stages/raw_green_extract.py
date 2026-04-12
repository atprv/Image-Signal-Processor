import torch
import torch.nn as nn
import torch.nn.functional as F


class RawGreenExtract(nn.Module):
    """
    Extract the green channel from Bayer RGGB and scale it to full resolution.
    This is used for raw-guided Y blending.
    """

    def forward(self, bayer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bayer: Bayer RGGB image [H, W], float32 in [0, 1]
        Returns:
            torch.Tensor: Green channel [H, W], float32 in [0, 1]
        """
        gr = bayer[::2, 1::2]
        gb = bayer[1::2, ::2]
        raw_g_half = 0.5 * (gr + gb)

        max_val = raw_g_half.max()
        if max_val > 0:
            raw_g_half = raw_g_half / max_val

        raw_g_4d = raw_g_half.unsqueeze(0).unsqueeze(0)
        raw_green = (
            F.interpolate(
                raw_g_4d,
                size=(bayer.shape[0], bayer.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        return raw_green
