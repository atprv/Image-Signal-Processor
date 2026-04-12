import torch
import torch.nn as nn


class AWB(nn.Module):
    """
    Gray-world auto white balance for Bayer images.
    """

    def __init__(
        self, max_gain: float = 4.0, lum_mask_low: float = 0.0, lum_mask_high: float = 1.0
    ):
        """
        Args:
            max_gain: Maximum allowed gain
            lum_mask_low: Lower luminance threshold for AWB stats
            lum_mask_high: Upper luminance threshold
        """
        super().__init__()

        self.register_buffer("max_gain", torch.tensor(max_gain, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(1.0, dtype=torch.float32))
        self.lum_mask_low = lum_mask_low
        self.lum_mask_high = lum_mask_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply AWB to a Bayer image.

        Args:
            x: Bayer RGGB image, shape [H, W], float32 in [0, 0xFFFFFF]

        Returns:
            torch.Tensor: AWB-corrected image, shape [H, W], float32 in [0, 0xFFFFFF]
        """
        r = x[::2, ::2]
        gr = x[::2, 1::2]
        gb = x[1::2, ::2]
        b = x[1::2, 1::2]

        use_lum_mask = self.lum_mask_low > 0.0 or self.lum_mask_high < 1.0
        if use_lum_mask:
            g_avg = 0.5 * (gr + gb)
            max_val = g_avg.max()
            if max_val > 0:
                g_norm = g_avg / max_val
            else:
                g_norm = g_avg
            lum_mask = (g_norm >= self.lum_mask_low) & (g_norm <= self.lum_mask_high)

            r_masked = r[lum_mask]
            gr_masked = gr[lum_mask]
            gb_masked = gb[lum_mask]
            b_masked = b[lum_mask]

            if r_masked.numel() == 0:
                r_masked, gr_masked, gb_masked, b_masked = r, gr, gb, b
        else:
            r_masked, gr_masked, gb_masked, b_masked = r, gr, gb, b

        r_ref = r_masked.mean()
        g_ref = 0.5 * (gr_masked.mean() + gb_masked.mean())
        b_ref = b_masked.mean()

        r_gain = g_ref / (r_ref + self.eps)
        b_gain = g_ref / (b_ref + self.eps)

        r_gain = torch.clamp(r_gain, 1.0 / self.max_gain, self.max_gain)
        b_gain = torch.clamp(b_gain, 1.0 / self.max_gain, self.max_gain)

        output = x.clone()
        output[::2, ::2] *= r_gain
        output[1::2, 1::2] *= b_gain

        return torch.clamp(output, 0, 0xFFFFFF)
