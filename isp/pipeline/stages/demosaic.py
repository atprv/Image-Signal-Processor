import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(nn.Module):
    """
    Demosaic a Bayer image with the Malvar-He-Cutler algorithm.
    """

    def __init__(self):
        super().__init__()

        # Malvar-He-Cutler filters

        G_at_RB = (
            torch.tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=torch.float32,
            )
            / 8
        )

        R_at_G_Rrow = (
            torch.tensor(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ],
                dtype=torch.float32,
            )
            / 8
        )

        R_at_G_Brow = (
            torch.tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, 4, -1, 0],
                    [0.5, 0, 5, 0, 0.5],
                    [0, -1, 4, -1, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=torch.float32,
            )
            / 8
        )

        R_at_B = (
            torch.tensor(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ],
                dtype=torch.float32,
            )
            / 8
        )

        batched_kernels = torch.stack(
            [
                G_at_RB.unsqueeze(0),
                R_at_G_Rrow.unsqueeze(0),
                R_at_G_Brow.unsqueeze(0),
                R_at_B.unsqueeze(0),
            ],
            dim=0,
        )

        self.register_buffer("batched_kernels", batched_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply demosaicing to a Bayer image.

        Args:
            x: Bayer RGGB image, shape [H, W], float32 in [0, 0xFFFFFF]

        Returns:
            torch.Tensor: RGB image, shape [H, W, 3], float32 in [0, 0xFFFFFF]
        """
        H, W = x.shape

        R = torch.zeros_like(x)
        G = torch.zeros_like(x)
        B = torch.zeros_like(x)

        R[::2, ::2] = x[::2, ::2]
        G[::2, 1::2] = x[::2, 1::2]
        G[1::2, ::2] = x[1::2, ::2]
        B[1::2, 1::2] = x[1::2, 1::2]

        x_4d = x.unsqueeze(0).unsqueeze(0)

        x_padded = F.pad(x_4d, (2, 2, 2, 2), mode="reflect")

        all_results = F.conv2d(x_padded, self.batched_kernels)

        G_interp = all_results[0, 0:1, :, :]
        R_g_r = all_results[0, 1:2, :, :]
        R_g_b = all_results[0, 2:3, :, :]
        R_b = all_results[0, 3:4, :, :]

        G[::2, ::2] = G_interp[0, ::2, ::2]
        G[1::2, 1::2] = G_interp[0, 1::2, 1::2]

        R[::2, 1::2] = R_g_r[0, ::2, 1::2]
        R[1::2, ::2] = R_g_b[0, 1::2, ::2]
        R[1::2, 1::2] = R_b[0, 1::2, 1::2]

        B[1::2, ::2] = R_g_r[0, 1::2, ::2]
        B[::2, 1::2] = R_g_b[0, ::2, 1::2]
        B[::2, ::2] = R_b[0, ::2, ::2]

        rgb = torch.stack([R, G, B], dim=-1)

        return torch.clamp(rgb, 0, 0xFFFFFF)
