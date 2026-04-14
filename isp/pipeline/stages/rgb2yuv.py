import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBtoYUV(nn.Module):
    """
    Convert RGB to NV12 YUV (4:2:0).
    """

    def __init__(self, raw_y_blend: float = 0.0, raw_y_blur_radius: int = 8):
        """
        Args:
            raw_y_blend: Blend factor for RAW high-frequency detail in Y
            raw_y_blur_radius: Box-filter radius for RAW base/detail split
        """
        super().__init__()

        rgb_to_y = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        rgb_to_u = torch.tensor([-0.1146, -0.3854, 0.5000], dtype=torch.float32)
        rgb_to_v = torch.tensor([0.5000, -0.4542, -0.0458], dtype=torch.float32)

        rgb2yuv_matrix = torch.stack([rgb_to_y, rgb_to_u, rgb_to_v], dim=0)
        self.register_buffer("rgb2yuv_matrix", rgb2yuv_matrix)

        self.raw_y_blend = raw_y_blend
        self.raw_y_blur_radius = raw_y_blur_radius

        if raw_y_blend > 0:
            ks = 2 * raw_y_blur_radius + 1
            box_1d = torch.ones(1, 1, 1, ks, dtype=torch.float32) / ks
            self.register_buffer("box_h", box_1d)
            self.register_buffer("box_v", box_1d.transpose(2, 3))

    def _separable_box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast separable box filter.

        Args:
            x: [1, 1, H, W]
        """
        r = self.raw_y_blur_radius
        x_padded = F.pad(x, (r, r, 0, 0), mode="reflect")
        x_h = F.conv2d(x_padded, self.box_h)
        x_h_padded = F.pad(x_h, (0, 0, r, r), mode="reflect")
        return F.conv2d(x_h_padded, self.box_v)

    def _compute_yuv420(
        self, x: torch.Tensor, raw_green: torch.Tensor = None, full_blend: float = 0.0
    ):
        """
        Shared float computation for both forward and forward_components.

        Returns:
            Y: float32 [H, W] in [0, 1]
            U_420: float32 [H/2, W/2] centred around 0
            V_420: float32 [H/2, W/2] centred around 0
        """
        yuv = x @ self.rgb2yuv_matrix.T

        Y = yuv[..., 0]
        U = yuv[..., 1]
        V = yuv[..., 2]

        if self.raw_y_blend > 0 and raw_green is not None:
            rg = raw_green
            y_mean = Y.mean()
            y_std = Y.std() + 1e-6
            rg_mean = rg.mean()
            rg_std = rg.std() + 1e-6

            rg_matched = (rg - rg_mean) / rg_std * y_std + y_mean

            rg_4d = rg_matched.unsqueeze(0).unsqueeze(0)
            rg_base = self._separable_box_filter(rg_4d).squeeze(0).squeeze(0)
            rg_detail = rg_matched - rg_base

            y_4d = Y.unsqueeze(0).unsqueeze(0)
            y_base = self._separable_box_filter(y_4d).squeeze(0).squeeze(0)
            y_detail = Y - y_base

            blended_detail = (1.0 - self.raw_y_blend) * y_detail + self.raw_y_blend * rg_detail
            Y = y_base + blended_detail
            Y = Y.clamp(0.0, 1.0)

        if full_blend > 0 and raw_green is not None:
            rg = raw_green
            y_mean = Y.mean()
            y_std = Y.std() + 1e-6
            rg_mean = rg.mean()
            rg_std = rg.std() + 1e-6
            rg_matched = (rg - rg_mean) / rg_std * y_std + y_mean
            Y = (1.0 - full_blend) * Y + full_blend * rg_matched
            Y = Y.clamp(0.0, 1.0)

        U_4d = U.unsqueeze(0).unsqueeze(0)
        V_4d = V.unsqueeze(0).unsqueeze(0)
        U_420 = F.avg_pool2d(U_4d, kernel_size=2, stride=2).squeeze()
        V_420 = F.avg_pool2d(V_4d, kernel_size=2, stride=2).squeeze()

        return Y, U_420, V_420

    def forward_components(
        self, x: torch.Tensor, raw_green: torch.Tensor = None, full_blend: float = 0.0
    ) -> dict:
        """
        Differentiable float path for training.

        Returns:
            dict with:
                y:  float32 [1, 1, H, W] in [0, 1]
                uv: float32 [1, 2, H/2, W/2] in [0, 1] (U, V shifted to unsigned)
        """
        Y, U_420, V_420 = self._compute_yuv420(x, raw_green, full_blend)

        U_unsigned = (U_420 + 128.0 / 255.0).clamp(0.0, 1.0)
        V_unsigned = (V_420 + 128.0 / 255.0).clamp(0.0, 1.0)

        y_out = Y.unsqueeze(0).unsqueeze(0)
        uv_out = torch.stack([U_unsigned, V_unsigned], dim=0).unsqueeze(0)

        return {"y": y_out, "uv": uv_out}

    def forward(
        self, x: torch.Tensor, raw_green: torch.Tensor = None, full_blend: float = 0.0
    ) -> torch.Tensor:
        """
        Convert RGB to NV12 YUV.

        Args:
            x: RGB image, shape [H, W, 3], float32 in [0, 1]
            raw_green: Optional RAW green channel [H, W], float32 in [0, 1]
            full_blend: Full Y blend factor with raw_green in [0.0, 1.0]

        Returns:
            torch.Tensor: NV12 YUV frame as a 1D uint8 tensor
        """
        H, W, _ = x.shape
        Y, U_420, V_420 = self._compute_yuv420(x, raw_green, full_blend)

        Y_uint8 = (Y * 255.0).clamp(0, 255).byte()
        U_420_uint8 = (U_420 * 255.0 + 128.0).clamp(0, 255).byte()
        V_420_uint8 = (V_420 * 255.0 + 128.0).clamp(0, 255).byte()

        yuv_size = H * W + 2 * (H // 2) * (W // 2)
        yuv_out = torch.empty(yuv_size, dtype=torch.uint8, device=x.device)
        yuv_out[: H * W] = Y_uint8.flatten()
        uv_interleaved = torch.stack([U_420_uint8, V_420_uint8], dim=-1).flatten()
        yuv_out[H * W :] = uv_interleaved

        return yuv_out
