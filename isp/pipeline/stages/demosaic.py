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
        self._mask_cache: dict[
            tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, ...]
        ] = {}

    @staticmethod
    def _is_compiling() -> bool:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())
        dynamo = getattr(torch, "_dynamo", None)
        dynamo_is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(dynamo_is_compiling):
            return bool(dynamo_is_compiling())
        return False

    @staticmethod
    def _build_pattern_masks(
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_even = (torch.arange(H, device=device) % 2 == 0).to(dtype).unsqueeze(1)
        col_even = (torch.arange(W, device=device) % 2 == 0).to(dtype).unsqueeze(0)
        row_odd = 1.0 - row_even
        col_odd = 1.0 - col_even

        return (
            row_even * col_even,
            row_even * col_odd,
            row_odd * col_even,
            row_odd * col_odd,
        )

    def clear_runtime_cache(self) -> None:
        self._mask_cache.clear()

    def prime_runtime_cache(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build and store parity masks outside torch.compile()."""
        key = (H, W, device, dtype)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_pattern_masks(H, W, device, dtype)

    def _get_pattern_masks(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache the RGGB parity masks."""
        key = (H, W, device, dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        if self._is_compiling():
            return self._build_pattern_masks(H, W, device, dtype)

        masks = self._build_pattern_masks(H, W, device, dtype)
        self._mask_cache[key] = masks
        return masks

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

        m_R, m_Gr, m_Gb, m_B = self._get_pattern_masks(H, W, x.device, x.dtype)

        R = m_R * x + m_Gr * R_g_r + m_Gb * R_g_b + m_B * R_b
        G = (m_R + m_B) * G_interp + (m_Gr + m_Gb) * x
        B = m_B * x + m_Gb * R_g_r + m_Gr * R_g_b + m_R * R_b

        rgb = torch.stack([R, G, B], dim=-1)
        return torch.clamp(rgb, 0.0, 1.0)
