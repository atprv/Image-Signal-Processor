import torch
import torch.nn as nn


class AWB(nn.Module):
    """
    Gray-world auto white balance for Bayer images.
    """

    def __init__(
        self,
        max_gain: float = 4.0,
        lum_mask_low: float = 0.0,
        lum_mask_high: float = 1.0,
        mask_temperature: float = 50.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            max_gain: Maximum allowed gain
            lum_mask_low: Lower luminance threshold for AWB stats
            lum_mask_high: Upper luminance threshold
            mask_temperature: Sigmoid steepness for soft masking
            eps: Small constant for stable normalization and gain computation
        """
        super().__init__()

        self.register_buffer("max_gain", torch.tensor(max_gain, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))
        self.lum_mask_low = lum_mask_low
        self.lum_mask_high = lum_mask_high
        self.mask_temperature = mask_temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply AWB to a Bayer image.

        Args:
            x: Bayer RGGB image, shape [H, W], float32

        Returns:
            torch.Tensor: AWB-corrected image, shape [H, W], float32
        """
        H, W = x.shape

        r = x[::2, ::2]
        gr = x[::2, 1::2]
        gb = x[1::2, ::2]
        b = x[1::2, 1::2]

        use_lum_mask = self.lum_mask_low > 0.0 or self.lum_mask_high < 1.0

        if use_lum_mask:
            g_avg = 0.5 * (gr + gb)
            max_val = g_avg.max().clamp(min=self.eps)
            g_norm = g_avg / max_val

            t = self.mask_temperature
            low_mask = torch.sigmoid(t * (g_norm - self.lum_mask_low))
            high_mask = torch.sigmoid(t * (self.lum_mask_high - g_norm))
            soft_mask = low_mask * high_mask

            mask_sum = soft_mask.sum().clamp(min=self.eps)
            r_ref = (r * soft_mask).sum() / mask_sum
            g_ref = 0.5 * ((gr * soft_mask).sum() / mask_sum + (gb * soft_mask).sum() / mask_sum)
            b_ref = (b * soft_mask).sum() / mask_sum
        else:
            r_ref = r.mean()
            g_ref = 0.5 * (gr.mean() + gb.mean())
            b_ref = b.mean()

        r_gain = torch.clamp(g_ref / (r_ref + self.eps), 1.0 / self.max_gain, self.max_gain)
        b_gain = torch.clamp(g_ref / (b_ref + self.eps), 1.0 / self.max_gain, self.max_gain)

        r_wb = r * r_gain
        b_wb = b * b_gain

        row_even = torch.stack([r_wb, gr], dim=-1).reshape(-1, W)
        row_odd = torch.stack([gb, b_wb], dim=-1).reshape(-1, W)
        output = torch.stack([row_even, row_odd], dim=1).reshape(H, W)

        return output.clamp(0.0, 1.0)
