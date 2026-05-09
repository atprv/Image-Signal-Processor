import torch
import torch.nn as nn
import torch.nn.functional as F


class LTM(nn.Module):
    """
    Local tone mapping with a fast guided filter.
    """

    def __init__(
        self,
        a: float = 0.7,
        b: float = 0.0,
        radius: int = 8,
        eps: float = 1e-3,
        downsample_factor: float = 0.5,
        target_mean: float = 0.0,
        detail_gain: float = 1.0,
        detail_threshold: float = 0.0,
    ):
        """
        Args:
            a: Dynamic-range compression factor (trainable)
            b: Brightness shift in the log domain (trainable)
            radius: Guided-filter radius (structural, integer)
            eps: Guided-filter regularization (trainable)
            downsample_factor: Downsample factor for the fast path (structural)
            target_mean: Output mean target in [0, 1] (trainable)
            detail_gain: Gain for local detail (trainable)
            detail_threshold: Noise threshold for log-detail suppression (trainable)
        """
        super().__init__()

        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float32))
        self.target_mean = nn.Parameter(torch.tensor(float(target_mean), dtype=torch.float32))
        self.detail_gain = nn.Parameter(torch.tensor(float(detail_gain), dtype=torch.float32))
        self.detail_threshold = nn.Parameter(
            torch.tensor(float(detail_threshold), dtype=torch.float32)
        )

        self.radius = radius
        self.downsample_factor = downsample_factor

        self.register_buffer("_param_floor", torch.tensor(1e-8, dtype=torch.float32))
        eps_value = max(float(eps), float(self._param_floor.item()))
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps_value, dtype=torch.float32)))
        self.register_buffer("eps_log", torch.tensor(1e-6, dtype=torch.float32))
        self.register_buffer("eps_scale", torch.tensor(1e-4, dtype=torch.float32))

        self._rebuild_box_filters()

    @property
    def eps(self) -> torch.Tensor:
        """Positive guided-filter regularizer."""
        return torch.exp(self.log_eps).clamp(min=self._param_floor)

    def set_eps(self, eps: float) -> None:
        eps_value = max(float(eps), float(self._param_floor.item()))
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
                ).clamp(min=self._param_floor)
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

    def _effective_radius(self) -> int:
        """Radius used inside the fast low-resolution path."""
        radius = int(self.radius)
        if radius <= 0:
            return 0
        if self.downsample_factor < 1.0:
            return max(1, int(round(radius * self.downsample_factor)))
        return radius

    def _rebuild_box_filters(self):
        """Rebuild separable box filters after radius/downsample changes."""
        effective_radius = self._effective_radius()
        kernel_size = 2 * effective_radius + 1
        device = self.log_eps.device
        box_1d = torch.ones(1, 1, 1, kernel_size, dtype=torch.float32, device=device) / kernel_size
        self.register_buffer("box_h", box_1d)
        self.register_buffer("box_v", box_1d.transpose(2, 3))
        self.pad = effective_radius

    def _separable_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast box filter via separable convolution.

        Args:
            x: Input tensor [1, 1, H, W]

        Returns:
            torch.Tensor: Filtered tensor [1, 1, H, W]
        """
        x_padded_h = F.pad(x, (self.pad, self.pad, 0, 0), mode="reflect")
        x_h = F.conv2d(x_padded_h, self.box_h)
        x_padded_v = F.pad(x_h, (0, 0, self.pad, self.pad), mode="reflect")
        return F.conv2d(x_padded_v, self.box_v)

    def _fast_guided_filter(self, guide: torch.Tensor) -> torch.Tensor:
        """
        Fast guided filter with separable box filters.

        Args:
            guide: Input image [H, W]

        Returns:
            torch.Tensor: Filtered image [H, W]
        """
        guide_4d = guide.unsqueeze(0).unsqueeze(0)

        mean_I = self._separable_box_filter(guide_4d)
        mean_II = self._separable_box_filter(guide_4d * guide_4d)

        var_I = (mean_II - mean_I * mean_I).clamp(min=0.0)

        a_local = var_I / (var_I + self.eps)
        b_local = mean_I - a_local * mean_I

        mean_a = self._separable_box_filter(a_local)
        mean_b = self._separable_box_filter(b_local)

        out = mean_a * guide_4d + mean_b

        return out.squeeze(0).squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply local tone mapping to an RGB image.

        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 1]

        Returns:
            torch.Tensor: Tone-mapped image, shape [H, W, 3], float32 in [0, 1]
        """
        H, W = x.shape[0], x.shape[1]

        R = x[..., 0]
        G = x[..., 1]
        B = x[..., 2]

        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

        Y_safe = torch.clamp(Y, min=self.eps_log)
        Y_log = torch.log2(Y_safe)

        if self.downsample_factor < 1.0:
            Y_log_4d = Y_log.unsqueeze(0).unsqueeze(0)
            Y_log_down = F.interpolate(
                Y_log_4d,
                scale_factor=self.downsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            Y_log_down_2d = Y_log_down.squeeze(0).squeeze(0)

            Y_base_down = self._fast_guided_filter(Y_log_down_2d)

            Y_base_4d = Y_base_down.unsqueeze(0).unsqueeze(0)
            Y_base_up = F.interpolate(Y_base_4d, size=(H, W), mode="bilinear", align_corners=False)
            Y_base = Y_base_up.squeeze(0).squeeze(0)
        else:
            Y_base = self._fast_guided_filter(Y_log)

        Y_detail = Y_log - Y_base

        thresh_safe = torch.clamp(self.detail_threshold, min=0.0)
        abs_d = Y_detail.abs()
        gated = (abs_d - thresh_safe).clamp(min=0.0)
        mask = gated / (gated + thresh_safe + 1e-8)
        Y_detail = Y_detail * mask

        Y_base_tm = self.a * Y_base + self.b
        Y_tm = Y_base_tm + self.detail_gain * Y_detail
        Y_tm = torch.clamp(Y_tm, min=-50.0, max=10.0)
        Y_out = torch.pow(2.0, Y_tm)

        scale = Y_out / (Y + self.eps_scale)

        output = torch.stack([R * scale, G * scale, B * scale], dim=-1)
        output = torch.clamp(output, 0.0, 1.0)

        current_mean = output.mean().clamp(min=1e-6)
        gate = torch.sigmoid(self.target_mean * 200.0 - 5.0)
        target_effective = self.target_mean * gate + current_mean * (1.0 - gate)
        output = torch.clamp(output * (target_effective / current_mean), 0.0, 1.0)

        return output
