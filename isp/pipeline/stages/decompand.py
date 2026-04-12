import torch
import torch.nn as nn


class DecompandBlackLevel(nn.Module):
    """
    Decompand a 12-bit frame and subtract black level.
    """

    def __init__(self, decompand_config: dict):
        """
        Args:
            decompand_config: Decompanding and black-level configuration
        """
        super().__init__()

        compand_knee = decompand_config["compand_knee"]
        compand_lut = decompand_config["compand_lut"]
        black_level = int(decompand_config["black_level"])

        device = compand_knee.device

        lut_full = torch.zeros(4096, dtype=torch.float32, device=device)

        for i in range(len(compand_knee) - 1):
            start_idx = compand_lut[i].item()
            end_idx = compand_lut[i + 1].item()
            start_val = compand_knee[i].item()
            end_val = compand_knee[i + 1].item()

            num_points = end_idx - start_idx
            if num_points > 0:
                segment_values = torch.linspace(
                    start_val, end_val, steps=num_points, dtype=torch.float32, device=device
                )
                lut_full[start_idx:end_idx] = segment_values

        lut_full[compand_lut[-1].item() :] = float(compand_knee[-1].item())

        lut_full = lut_full - float(black_level)

        self.register_buffer("lut", lut_full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply decompanding and black-level subtraction.

        Args:
            x: Input 12-bit frame, shape [H, W]

        Returns:
            torch.Tensor: Processed frame, shape [H, W], float32 in [0, 0xFFFFFF]
        """
        x_int = x.to(torch.int32)

        x_clipped = torch.clamp(x_int, 0, 4095)
        output = self.lut[x_clipped]

        return output
