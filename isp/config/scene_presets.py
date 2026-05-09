"""
Scene-specific ISP presets used for evaluation and benchmarking.
"""

from __future__ import annotations

from copy import deepcopy

SCENE_ISP_PARAMS: dict[str, dict[str, float | int]] = {
    "day": {
        "denoise_eps": 1e-12,
        "ltm_a": 0.5,
        "ltm_detail_gain": 30,
        "ltm_detail_threshold": 0.35,
        "hist_target_mean": 0.445,
        "hist_target_std": 0.162,
        "post_denoise_radius": 4,
        "post_denoise_eps": 0.001,
        "raw_y_full_blend": 0.4,
        "sharp_amount": 0.3,
    },
    "night": {
        "denoise_eps": 1e-12,
        "ltm_a": 0.3,
        "ltm_detail_gain": 8,
        "ltm_detail_threshold": 0.4,
        "sharp_amount": 0.8,
    },
    "tunnel": {},
}


def get_scene_isp_params(scene_name: str) -> dict[str, float | int]:
    """
    Return a defensive copy of the ISP preset for one scene.
    """
    key = scene_name.strip().lower()
    if key not in SCENE_ISP_PARAMS:
        available = ", ".join(sorted(SCENE_ISP_PARAMS))
        raise KeyError(f"Unknown scene '{scene_name}'. Available: {available}")
    return deepcopy(SCENE_ISP_PARAMS[key])
