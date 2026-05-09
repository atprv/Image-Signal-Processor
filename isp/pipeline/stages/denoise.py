import torch
import torch.nn as nn
import torch.nn.functional as F


class BayerDenoise(nn.Module):
    """
    Bayer denoise with a guided filter.
    """

    _EPS_FLOOR = 1e-20

    def __init__(self, radius: int = 2, eps: float = 100.0):
        """
        Args:
            radius: Filter radius (structural)
            eps: Guided-filter regularization (trainable)
        """
        super().__init__()

        self.radius = radius
        eps_value = max(float(eps), self._EPS_FLOOR)
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps_value, dtype=torch.float32)))

        kernel_size = 2 * radius + 1
        box_kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32) / (
            kernel_size**2
        )
        self.register_buffer("box_kernel", box_kernel)
        box_1d = torch.ones(1, 1, 1, kernel_size, dtype=torch.float32) / kernel_size
        self.register_buffer("box_h", box_1d)
        self.register_buffer("box_v", box_1d.transpose(2, 3))

        self.pad = radius

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

    def _fast_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast box filter via convolution.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            torch.Tensor: Filtered tensor [B, 1, H, W]
        """
        x_padded = F.pad(x, (self.pad, self.pad, 0, 0), mode="reflect")
        x_h = F.conv2d(x_padded, self.box_h)
        x_h_padded = F.pad(x_h, (0, 0, self.pad, self.pad), mode="reflect")
        return F.conv2d(x_h_padded, self.box_v)

    def _guided_filter_batch(self, I_batch: torch.Tensor) -> torch.Tensor:
        """
        Guided filter for a batch of channels.

        Args:
            I_batch: Input batch [B, 1, H, W]

        Returns:
            torch.Tensor: Filtered batch [B, 1, H, W]
        """
        mean_I = self._fast_box_filter(I_batch)
        mean_II = self._fast_box_filter(I_batch * I_batch)

        var_I = (mean_II - mean_I * mean_I).clamp(min=0.0)

        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        mean_a = self._fast_box_filter(a)
        mean_b = self._fast_box_filter(b)

        out = mean_a * I_batch + mean_b

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoise to a Bayer image.

        Args:
            x: Bayer RGGB image, shape [H, W], float32 in [0, 1]

        Returns:
            torch.Tensor: Denoised image, shape [H, W], float32 in [0, 1]
        """
        r = x[::2, ::2]
        gr = x[::2, 1::2]
        gb = x[1::2, ::2]
        b = x[1::2, 1::2]

        channels = torch.stack([r, gr, gb, b], dim=0).unsqueeze(1)

        filtered_batch = self._guided_filter_batch(channels)

        r_filtered = filtered_batch[0, 0]
        gr_filtered = filtered_batch[1, 0]
        gb_filtered = filtered_batch[2, 0]
        b_filtered = filtered_batch[3, 0]

        output = torch.empty_like(x, dtype=torch.float32)
        output[::2, ::2] = r_filtered
        output[::2, 1::2] = gr_filtered
        output[1::2, ::2] = gb_filtered
        output[1::2, 1::2] = b_filtered

        return output.clamp(0.0, 1.0)
