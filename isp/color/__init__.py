"""
Color-space and pixel-format conversions used across the ISP / CNN / eval code.

All tensors are assumed to be in layout [B, C, H, W] unless noted otherwise.
Float tensors are expected in [0, 1] unless a function name explicitly says
otherwise.
"""

from .conversions import (
    float_y_to_uint8,
    normalize_uv_planes,
    normalize_y_plane,
    nv12_uint8_to_rgb_bt709_full,
    unpack_nv12_buffer,
    yuv420_to_rgb_bt709_full,
    yuv420_to_yuv444,
    yuv444_to_yuv420,
)

__all__ = [
    "float_y_to_uint8",
    "normalize_uv_planes",
    "normalize_y_plane",
    "nv12_uint8_to_rgb_bt709_full",
    "unpack_nv12_buffer",
    "yuv420_to_rgb_bt709_full",
    "yuv420_to_yuv444",
    "yuv444_to_yuv420",
]
