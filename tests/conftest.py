from pathlib import Path

import pytest
import toml


def build_config(width: int = 8, height: int = 8, emb_lines: tuple[int, int] = (0, 0)) -> dict:
    """Create a minimal camera config suitable for tests."""
    return {
        "img": {
            "width": width,
            "height": height,
            "emb_lines": list(emb_lines),
        },
        "decompanding": {
            "black_level": 0,
            "compand_knee": [0, 16777215],
            "compand_lut": [0, 4095],
        },
        "ccm": {
            "ccm_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        },
    }


@pytest.fixture
def minimal_config_dict() -> dict:
    """Return a small neutral camera config for tests."""
    return build_config()


@pytest.fixture
def minimal_config_path(tmp_path: Path, minimal_config_dict: dict) -> Path:
    """Write a minimal camera config to a temporary TOML file."""
    config_path = tmp_path / "camera.toml"
    config_path.write_text(toml.dumps(minimal_config_dict), encoding="utf-8")
    return config_path
