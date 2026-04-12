import torch

from isp.config import read_config


def test_read_config_converts_arrays_to_tensors(minimal_config_path):
    config = read_config(str(minimal_config_path), device="cpu")

    assert config["decompanding"]["compand_knee"].dtype == torch.int32
    assert config["decompanding"]["compand_lut"].dtype == torch.int32
    assert config["ccm"]["ccm_matrix"].dtype == torch.float32
    assert config["decompanding"]["compand_knee"].device.type == "cpu"
    assert config["ccm"]["ccm_matrix"].shape == (3, 3)
