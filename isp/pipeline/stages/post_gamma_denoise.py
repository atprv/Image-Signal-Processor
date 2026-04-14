import torch
import torch.nn as nn
import torch.nn.functional as F


class PostGammaDenoise(nn.Module):
    """
    Guided-filter denoise on luma in the perceptual domain.
    """

    def __init__(self, radius: int = 0, eps: float = 0.005):
        """
        Args:
            radius: Guided-filter radius; 0 disables the stage
            eps: Guided-filter regularization
        """
        super().__init__()
        self.radius = radius
        self.eps = eps

        if radius > 0:
            ks = 2 * radius + 1
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32) / ks
            self.register_buffer("box_h", box_1d)
            self.register_buffer("box_v", box_1d.transpose(2, 3))

    def _box_filter(self, t: torch.Tensor) -> torch.Tensor:
        """
        Separable box filter.

        Args:
            t: [1, 1, H, W]
        """
        r = self.radius
        t = F.pad(t, (r, r, r, r), mode="reflect")
        t = F.conv2d(t, self.box_h)
        t = F.conv2d(t, self.box_v)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Denoised RGB image [H, W, 3], float32 in [0, 1]
        """
        if self.radius <= 0:
            return x

        Y_lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        guide = Y_lum.unsqueeze(0).unsqueeze(0)

        mean_I = self._box_filter(guide)
        mean_II = self._box_filter(guide * guide)
        var_I = mean_II - mean_I * mean_I

        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        Y_smooth = (mean_a * guide + mean_b).squeeze(0).squeeze(0)

        scale = (Y_smooth / Y_lum.clamp(min=1e-6)).unsqueeze(-1)
        return (x * scale).clamp(0.0, 1.0)
