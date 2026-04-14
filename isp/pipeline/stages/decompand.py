import torch
import torch.nn as nn


class DecompandBlackLevel(nn.Module):
    """
    Decompand a 12-bit frame, subtract black level, and normalize to [0, 1].
    """

    def __init__(self, decompand_config: dict):
        """
        Args:
            decompand_config: Decompanding and black-level configuration
        """
        super().__init__()

        compand_knee = decompand_config["compand_knee"]
        compand_lut = decompand_config["compand_lut"]
        black_level = float(decompand_config["black_level"])

        device = compand_knee.device

        lut_full = torch.zeros(4096, dtype=torch.float32, device=device)

        for i in range(len(compand_knee) - 1):
            start_idx = int(compand_lut[i].item())
            end_idx = int(compand_lut[i + 1].item())
            start_val = compand_knee[i].item()
            end_val = compand_knee[i + 1].item()

            num_points = end_idx - start_idx
            if num_points > 0:
                segment_values = torch.linspace(
                    start_val,
                    end_val,
                    steps=num_points,
                    dtype=torch.float32,
                    device=device,
                )
                lut_full[start_idx:end_idx] = segment_values

        lut_full[int(compand_lut[-1].item()) :] = float(compand_knee[-1].item())

        white_level = float(compand_knee[-1].item()) - black_level
        if white_level <= 0:
            raise ValueError(f"Expected positive white level, got {white_level}")

        output_scale = float(0xFFFFFF)
        lut_full = (lut_full - black_level) / output_scale

        self.register_buffer("lut", lut_full)
        self.register_buffer(
            "white_level",
            torch.tensor(white_level, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "output_scale",
            torch.tensor(output_scale, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "max_index",
            torch.tensor(float(lut_full.numel() - 1), dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply decompanding and black-level subtraction.

        Args:
            x: Input 12-bit frame, shape [H, W], uint16 or float tensor

        Returns:
            torch.Tensor: Processed frame, shape [H, W], float tensor in [0, 1]
        """
        return self.forward_unclamped(x).clamp(0.0, 1.0)

    def forward_unclamped(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal differentiable path without output clamping.
        """
        work_dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        x_f = x.to(device=self.lut.device, dtype=work_dtype).clamp(
            0.0, self.max_index.to(dtype=work_dtype)
        )
        lut = self.lut.to(dtype=work_dtype)

        idx_lo = x_f.detach().floor().long()
        idx_hi = (idx_lo + 1).clamp(max=int(self.max_index.item()))

        frac = x_f - idx_lo.to(dtype=work_dtype)

        y_lo = lut[idx_lo]
        y_hi = lut[idx_hi]

        return torch.lerp(y_lo, y_hi, frac)
