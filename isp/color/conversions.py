"""
Color-space and pixel-format conversions.

Canonical conventions used across the codebase:
- Y tensor:   [B, 1, H, W]
- UV tensor:  [B, 2, H/2, W/2]  (U first, V second)
- YUV444:     [B, 3, H, W]      (Y, U, V)
- RGB:        [B, 3, H, W]      in [0, 1]
"""

import torch
import torch.nn.functional as F


def normalize_y_plane(y: torch.Tensor) -> torch.Tensor:
    """uint8 Y plane -> float32 [0, 1]."""
    return y.to(torch.float32) / 255.0


def normalize_uv_planes(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """uint8 U/V planes -> float32 UV tensor [B, 2, H/2, W/2] in [0, 1]."""
    return torch.cat([u.to(torch.float32), v.to(torch.float32)], dim=1) / 255.0


def float_y_to_uint8(y_float: torch.Tensor) -> torch.Tensor:
    """float32 Y in [0, 1] -> uint8 (rounded, clamped)."""
    return (y_float * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)


def unpack_nv12_buffer(
    nv12: torch.Tensor, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack a flat NV12 buffer into separate Y, U, V tensors.

    Returns:
        y: [1, 1, H, W]
        u: [1, 1, H/2, W/2]
        v: [1, 1, H/2, W/2]
    """
    expected_size = width * height + width * height // 2
    nv12_flat = nv12.reshape(-1)

    if nv12_flat.numel() != expected_size:
        raise ValueError(f"Unexpected NV12 size: got {nv12_flat.numel()}, expected {expected_size}")

    y_size = width * height
    uv_height = height // 2
    uv_width = width // 2

    y = nv12_flat[:y_size].reshape(1, 1, height, width)
    uv = nv12_flat[y_size:].reshape(uv_height, uv_width, 2)

    u = uv[..., 0].unsqueeze(0).unsqueeze(0)
    v = uv[..., 1].unsqueeze(0).unsqueeze(0)

    return y, u, v


def yuv420_to_yuv444(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Upsample planar YUV420 (UV at half resolution) to YUV444 for per-pixel processing.
    """
    uv_up = F.interpolate(uv, size=y.shape[2:], mode="bilinear", align_corners=False)
    return torch.cat([y, uv_up], dim=1)


def yuv444_to_yuv420(yuv444: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample YUV444 chroma back to 4:2:0. Y is passed through unchanged.
    """
    if yuv444.ndim != 4 or yuv444.shape[1] != 3:
        raise ValueError(f"Expected YUV444 tensor [B, 3, H, W], got {tuple(yuv444.shape)}")

    y = yuv444[:, :1]
    uv444 = yuv444[:, 1:]

    if y.shape[2] % 2 != 0 or y.shape[3] % 2 != 0:
        raise ValueError(f"YUV420 requires even spatial size, got {tuple(y.shape[2:])}")

    uv420 = F.avg_pool2d(uv444, kernel_size=2, stride=2)
    return y, uv420


def yuv420_to_rgb_bt709_full(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Convert planar YUV420 (float32, [0, 1]) to RGB (float32, [0, 1]) using the
    BT.709 full-range matrix.

    Args:
        y:  [B, 1, H, W]    float in [0, 1]
        uv: [B, 2, H/2, W/2] float in [0, 1], U first, V second

    Returns:
        rgb: [B, 3, H, W] float in [0, 1]
    """
    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError(f"Expected Y tensor [B, 1, H, W], got {tuple(y.shape)}")
    if uv.ndim != 4 or uv.shape[1] != 2:
        raise ValueError(f"Expected UV tensor [B, 2, H/2, W/2], got {tuple(uv.shape)}")

    u = uv[:, 0:1] - (128.0 / 255.0)
    v = uv[:, 1:2] - (128.0 / 255.0)

    u_up = F.interpolate(u, size=y.shape[2:], mode="bilinear", align_corners=False)
    v_up = F.interpolate(v, size=y.shape[2:], mode="bilinear", align_corners=False)

    r = y + 1.5748 * v_up
    g = y - 0.1873 * u_up - 0.4681 * v_up
    b = y + 1.8556 * u_up

    return torch.cat(
        [
            torch.clamp(r, 0.0, 1.0),
            torch.clamp(g, 0.0, 1.0),
            torch.clamp(b, 0.0, 1.0),
        ],
        dim=1,
    ).contiguous()


def nv12_uint8_to_rgb_bt709_full(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Convert separate uint8 Y/U/V planes (as unpacked from NV12) to RGB in
    [0, 1]. Thin wrapper around yuv420_to_rgb_bt709_full for legacy eval
    code paths that hold Y and UV as independent tensors.

    Args:
        y: [B, 1, H, W]    uint8 or float
        u: [B, 1, H/2, W/2] uint8 or float
        v: [B, 1, H/2, W/2] uint8 or float
    """
    y_norm = y.to(torch.float32) / 255.0
    uv = torch.cat([u.to(torch.float32), v.to(torch.float32)], dim=1) / 255.0
    return yuv420_to_rgb_bt709_full(y_norm, uv)
