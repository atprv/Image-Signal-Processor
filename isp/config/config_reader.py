from typing import Any

import toml
import torch


def read_config(config_path: str, device: str = "cuda") -> dict[str, Any]:
    """
    Read a camera TOML config and prepare tensors on the target device.

    Args:
        config_path: Path to the TOML config file
        device: Target device for created tensors ('cuda' or 'cpu')

    Returns:
        dict: Camera configuration dictionary
    """
    with open(config_path) as f:
        config = toml.load(f)

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    device_obj = torch.device(device)

    if "decompanding" in config:
        if "compand_knee" in config["decompanding"]:
            config["decompanding"]["compand_knee"] = torch.tensor(
                config["decompanding"]["compand_knee"], dtype=torch.int32, device=device_obj
            )

        if "compand_lut" in config["decompanding"]:
            config["decompanding"]["compand_lut"] = torch.tensor(
                config["decompanding"]["compand_lut"], dtype=torch.int32, device=device_obj
            )

    if "ccm" in config and "ccm_matrix" in config["ccm"]:
        config["ccm"]["ccm_matrix"] = torch.tensor(
            config["ccm"]["ccm_matrix"], dtype=torch.float32, device=device_obj
        )

    return config
