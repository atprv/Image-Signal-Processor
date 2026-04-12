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
            a: Dynamic-range compression factor
            b: Brightness shift in the log domain
            radius: Guided-filter radius
            eps: Guided-filter regularization
            downsample_factor: Downsample factor for the fast path
            target_mean: Optional output mean target in [0, 1]
            detail_gain: Gain for local detail
            detail_threshold: Noise threshold for log-detail suppression
        """
        super().__init__()

        self.register_buffer("a", torch.tensor(a, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("detail_gain", torch.tensor(detail_gain, dtype=torch.float32))
        self.register_buffer(
            "detail_threshold", torch.tensor(detail_threshold, dtype=torch.float32)
        )
        self.radius = radius
        self.downsample_factor = downsample_factor
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

        self.register_buffer("eps_log", torch.tensor(1e-6, dtype=torch.float32))
        self.register_buffer("eps_scale", torch.tensor(1e-4, dtype=torch.float32))

        kernel_size = 2 * radius + 1

        box_1d = torch.ones(1, 1, 1, kernel_size, dtype=torch.float32) / kernel_size
        self.register_buffer("box_h", box_1d)  # Horizontal
        self.register_buffer("box_v", box_1d.transpose(2, 3))  # Vertical

        self.pad = radius

    def _separable_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast box filter via separable convolution.

        Args:
            x: Input tensor [1, 1, H, W]

        Returns:
            torch.Tensor: Filtered tensor [1, 1, H, W]
        """
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")

        x_h = F.conv2d(x_padded, self.box_h)
        x_hv = F.conv2d(x_h, self.box_v)

        return x_hv

    def _fast_guided_filter(self, image: torch.Tensor) -> torch.Tensor:
        """
        Fast guided filter with separable box filters.

        Args:
            image: Input image [H, W]

        Returns:
            torch.Tensor: Filtered image [H, W]
        """
        image_4d = image.unsqueeze(0).unsqueeze(0)

        mean_I = self._separable_box_filter(image_4d)
        mean_II = self._separable_box_filter(image_4d * image_4d)

        var_I = mean_II - mean_I * mean_I

        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I

        mean_a = self._separable_box_filter(a)
        mean_b = self._separable_box_filter(b)

        out = mean_a * image_4d + mean_b

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
                Y_log_4d, scale_factor=self.downsample_factor, mode="bilinear", align_corners=False
            )
            Y_log_down_2d = Y_log_down.squeeze(0).squeeze(0)

            Y_base_down = self._fast_guided_filter(Y_log_down_2d)

            Y_base_4d = Y_base_down.unsqueeze(0).unsqueeze(0)
            Y_base_up = F.interpolate(Y_base_4d, size=(H, W), mode="bilinear", align_corners=False)
            Y_base = Y_base_up.squeeze(0).squeeze(0)
        else:
            Y_base = self._fast_guided_filter(Y_log)

        Y_detail = Y_log - Y_base

        if self.detail_threshold > 0:
            abs_d = Y_detail.abs()
            mask = (abs_d - self.detail_threshold).clamp(min=0.0)
            mask = mask / (mask + self.detail_threshold + 1e-8)
            Y_detail = Y_detail * mask

        Y_base_tm = self.a * Y_base + self.b

        Y_tm = Y_base_tm + self.detail_gain * Y_detail
        Y_out = torch.pow(2.0, Y_tm)

        scale = Y_out / (Y + self.eps_scale)

        output = torch.empty_like(x)
        output[..., 0] = R * scale
        output[..., 1] = G * scale
        output[..., 2] = B * scale

        output = torch.clamp(output, 0.0, 1.0)

        if self.target_mean > 0:
            current_mean = output.mean()
            if current_mean > 1e-6:
                scale_factor = self.target_mean / current_mean
                output = torch.clamp(output * scale_factor, 0.0, 1.0)

        return output
