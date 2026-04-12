import torch
import torch.nn as nn


class HistogramNormalization(nn.Module):
    """
    Normalize luminance in the perceptual gamma-corrected domain.
    A power curve shifts the mean, then an optional linear step matches contrast.
    """

    def __init__(self, target_mean: float = 0.0, target_std: float = 0.0):
        """
        Args:
            target_mean: Target Y mean in [0, 1]
            target_std: Target Y std in [0, 1]
        """
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Normalized RGB image [H, W, 3], float32 in [0, 1]
        """
        if self.target_mean <= 0:
            return x

        Y_cur = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        cur_mean = Y_cur.mean()

        if cur_mean <= 1e-6:
            return x

        p = torch.log(torch.tensor(self.target_mean, device=x.device)) / torch.log(cur_mean)
        Y_safe = Y_cur.clamp(min=1e-6)
        Y_new = Y_safe.pow(p)

        if self.target_std > 0:
            cur_std_new = Y_new.std()
            if cur_std_new > 1e-6:
                std_gain = self.target_std / cur_std_new
                Y_mean_new = Y_new.mean()
                Y_new = (Y_new - Y_mean_new) * std_gain + Y_mean_new
                Y_new = Y_new.clamp(min=1e-6)

        rgb_scale = (Y_new / Y_cur.clamp(min=1e-6)).unsqueeze(-1)
        return (x * rgb_scale).clamp(0.0, 1.0)
