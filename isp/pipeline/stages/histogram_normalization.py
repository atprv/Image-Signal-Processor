import torch
import torch.nn as nn


class HistogramNormalization(nn.Module):
    """
    Normalize luminance in the perceptual gamma-corrected domain.
    """

    _TARGET_MEAN_FLOOR = 1e-3
    _CUR_MEAN_FLOOR = 1e-6
    _CUR_MEAN_CEIL = 1.0 - 1e-6

    def __init__(self, target_mean: float = 0.0, target_std: float = 0.0):
        """
        Args:
            target_mean: Target Y mean in [0, 1] (trainable)
            target_std: Target Y std in [0, 1] (trainable)
        """
        super().__init__()
        self.target_mean = nn.Parameter(torch.tensor(float(target_mean), dtype=torch.float32))
        self.target_std = nn.Parameter(torch.tensor(float(target_std), dtype=torch.float32))
        self._identity_fast_path = False
        self._refresh_inference_fast_flags()

    def _refresh_inference_fast_flags(self) -> None:
        self._identity_fast_path = bool(
            self.target_mean.detach().item() == 0.0 and self.target_std.detach().item() == 0.0
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self._refresh_inference_fast_flags()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [H, W, 3], float32 in [0, 1]
        Returns:
            torch.Tensor: Normalized RGB image [H, W, 3], float32 in [0, 1]
        """
        if (not self.training) and (not torch.is_grad_enabled()) and self._identity_fast_path:
            return x

        Y_cur = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        cur_mean = Y_cur.mean().clamp(min=self._CUR_MEAN_FLOOR, max=self._CUR_MEAN_CEIL)

        target_mean_safe = self.target_mean.clamp(min=self._TARGET_MEAN_FLOOR)
        p = torch.log(target_mean_safe) / torch.log(cur_mean)
        Y_safe = Y_cur.clamp(min=self._CUR_MEAN_FLOOR)
        Y_new_meaned = Y_safe.pow(p)

        mean_gate = torch.sigmoid(self.target_mean * 200.0 - 5.0)
        Y_after_mean = mean_gate * Y_new_meaned + (1.0 - mean_gate) * Y_cur

        cur_std_new = Y_after_mean.std().clamp(min=self._CUR_MEAN_FLOOR)
        target_std_safe = self.target_std.clamp(min=0.0)
        std_gain = target_std_safe / cur_std_new
        Y_mean_after = Y_after_mean.mean()
        Y_std_rescaled = (Y_after_mean - Y_mean_after) * std_gain + Y_mean_after
        Y_std_rescaled = Y_std_rescaled.clamp(min=self._CUR_MEAN_FLOOR)

        std_gate = torch.sigmoid(self.target_std * 200.0 - 5.0)
        Y_final = std_gate * Y_std_rescaled + (1.0 - std_gate) * Y_after_mean

        rgb_scale = (Y_final / Y_cur.clamp(min=self._CUR_MEAN_FLOOR)).unsqueeze(-1)
        return (x * rgb_scale).clamp(0.0, 1.0)
