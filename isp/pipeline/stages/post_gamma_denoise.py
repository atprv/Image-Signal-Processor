import torch
import torch.nn as nn
import torch.nn.functional as F


class PostGammaDenoise(nn.Module):
    """
    Guided-filter denoise on luma in the perceptual domain.
    """

    _EPS_FLOOR = 1e-8

    def __init__(self, radius: int = 0, eps: float = 0.005):
        """
        Args:
            radius: Guided-filter radius; 0 disables the stage (structural)
            eps: Guided-filter regularization (trainable)
        """
        super().__init__()
        self.radius = radius
        eps_value = max(float(eps), self._EPS_FLOOR)
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps_value, dtype=torch.float32)))

        if radius > 0:
            ks = 2 * radius + 1
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32) / ks
            self.register_buffer("box_h", box_1d)
            self.register_buffer("box_v", box_1d.transpose(2, 3))

    @property
    def eps(self) -> torch.Tensor:
        """Positive guided-filter regularizer."""
        return torch.exp(self.log_eps).clamp(min=self._EPS_FLOOR)

    def set_eps(self, eps: float) -> None:
        eps_value = max(float(eps), self._EPS_FLOOR)
        with torch.no_grad():
            self.log_eps.copy_(
                torch.log(
                    torch.tensor(
                        eps_value,
                        dtype=self.log_eps.dtype,
                        device=self.log_eps.device,
                    )
                )
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        log_key = prefix + "log_eps"
        legacy_key = prefix + "eps"
        fallback_eps = float(self.eps.detach().cpu().item())
        fallback_log_eps = float(self.log_eps.detach().cpu().item())

        if log_key in state_dict:
            log_eps = state_dict[log_key]
            if torch.is_tensor(log_eps) and not torch.isfinite(log_eps).all():
                state_dict[log_key] = torch.nan_to_num(
                    log_eps,
                    nan=fallback_log_eps,
                    posinf=fallback_log_eps,
                    neginf=fallback_log_eps,
                )
        elif legacy_key in state_dict:
            legacy_eps = state_dict.pop(legacy_key)
            if torch.is_tensor(legacy_eps):
                eps_value = torch.nan_to_num(
                    legacy_eps,
                    nan=fallback_eps,
                    posinf=fallback_eps,
                    neginf=fallback_eps,
                ).clamp(min=self._EPS_FLOOR)
                state_dict[log_key] = eps_value.log()

        state_dict.pop(legacy_key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _box_filter(self, t: torch.Tensor) -> torch.Tensor:
        """
        Separable box filter.

        Args:
            t: [1, 1, H, W]
        """
        r = self.radius
        t_padded_h = F.pad(t, (r, r, 0, 0), mode="reflect")
        t_h = F.conv2d(t_padded_h, self.box_h)
        t_padded_v = F.pad(t_h, (0, 0, r, r), mode="reflect")
        return F.conv2d(t_padded_v, self.box_v)

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

        var_I = (mean_II - mean_I * mean_I).clamp(min=0.0)

        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        Y_smooth = (mean_a * guide + mean_b).squeeze(0).squeeze(0)

        scale = (Y_smooth / Y_lum.clamp(min=1e-6)).unsqueeze(-1)
        return (x * scale).clamp(0.0, 1.0)
