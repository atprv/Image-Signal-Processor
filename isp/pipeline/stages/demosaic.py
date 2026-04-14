import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(nn.Module):
    """
    Demosaic a Bayer image with the Malvar-He-Cutler algorithm.
    """

    def __init__(self):
        super().__init__()

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
            x: Bayer RGGB image, shape [H, W], float32 in [0, 1]

        Returns:
            torch.Tensor: RGB image, shape [H, W, 3], float32 in [0, 1]
        """
        H, W = x.shape

        x_4d = x.unsqueeze(0).unsqueeze(0)
        x_padded = F.pad(x_4d, (2, 2, 2, 2), mode="reflect")
        all_results = F.conv2d(x_padded, self.batched_kernels)

        G_interp = all_results[0, 0]
        R_g_r = all_results[0, 1]
        R_g_b = all_results[0, 2]
        R_b = all_results[0, 3]

        row_even = (torch.arange(H, device=x.device) % 2 == 0).float().unsqueeze(1)
        col_even = (torch.arange(W, device=x.device) % 2 == 0).float().unsqueeze(0)
        row_odd = 1.0 - row_even
        col_odd = 1.0 - col_even

        m_R = row_even * col_even
        m_Gr = row_even * col_odd
        m_Gb = row_odd * col_even
        m_B = row_odd * col_odd

        R = m_R * x + m_Gr * R_g_r + m_Gb * R_g_b + m_B * R_b
        G = (m_R + m_B) * G_interp + (m_Gr + m_Gb) * x
        B = m_B * x + m_Gb * R_g_r + m_Gr * R_g_b + m_R * R_b

        rgb = torch.stack([R, G, B], dim=-1)
        return torch.clamp(rgb, 0.0, 1.0)
