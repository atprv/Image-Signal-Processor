import torch

from isp.config import read_config
from isp.pipeline import ISPPipeline
from scripts.generate_synthetic_raw import generate_synthetic_raw
from scripts.run_traditional_isp import process_video


def test_generate_synthetic_raw_produces_even_bayer_clip():
    clip = generate_synthetic_raw(width=8, height=8, frames=3)

    assert clip.shape == (3, 8, 8)
    assert clip.dtype.name == "uint16"
    assert clip.min() >= 0
    assert clip.max() <= 4095


def test_isp_pipeline_returns_nv12_buffer(minimal_config_path):
    config = read_config(str(minimal_config_path), device="cpu")
    pipeline = ISPPipeline(
        config,
        device="cpu",
        denoise_radius=1,
        ltm_radius=1,
        ltm_downsample=1.0,
        raw_y_blur_radius=1,
    )
    raw_frame = (torch.arange(64, dtype=torch.int32).reshape(8, 8) * 32).clamp(max=4095)
    raw_frame = raw_frame.to(dtype=torch.uint16)

    output = pipeline(raw_frame)

    assert output.dtype == torch.uint8
    assert output.numel() == 96


def test_process_video_runs_end_to_end_on_cpu(minimal_config_path, tmp_path):
    raw_clip = generate_synthetic_raw(width=8, height=8, frames=2)
    video_path = tmp_path / "input.bin"
    video_path.write_bytes(raw_clip.tobytes())
    output_path = tmp_path / "nested" / "output.yuv"

    stats = process_video(
        video_path=str(video_path),
        config_path=str(minimal_config_path),
        output_path=str(output_path),
        max_frames=1,
        verbose=False,
        device="cpu",
        denoise_radius=1,
        ltm_radius=1,
        ltm_downsample=1.0,
        raw_y_blur_radius=1,
    )

    assert stats["frames"] == 1
    assert output_path.exists()
    assert output_path.stat().st_size == 96
    assert stats["isp_fps"] >= 0
