import numpy as np
import torch

from isp.io.video_reader import RAWVideoReader
from isp.io.video_writer import AsyncYUVWriter, save_yuv
from isp.io.yuv_reader import NV12VideoReader
from tests.conftest import build_config


def test_raw_video_reader_crops_embedded_lines(tmp_path):
    config = build_config(width=4, height=6, emb_lines=(1, 1))
    frame = np.arange(24, dtype=np.uint16).reshape(6, 4)
    video_path = tmp_path / "sample.bin"
    video_path.write_bytes(frame.tobytes())

    with RAWVideoReader(str(video_path), config, device="cpu") as reader:
        cropped_frame, success = reader.read_frame()

    assert success is True
    assert tuple(cropped_frame.shape) == (4, 4)
    np.testing.assert_array_equal(cropped_frame.numpy(), frame[1:-1, :])


def test_nv12_video_reader_decodes_planes(tmp_path):
    width = 4
    height = 4
    y_plane = np.arange(width * height, dtype=np.uint8)
    uv_plane = np.array([100, 150, 101, 151, 102, 152, 103, 153], dtype=np.uint8)
    video_path = tmp_path / "sample_nv12.yuv"
    video_path.write_bytes(y_plane.tobytes() + uv_plane.tobytes())

    with NV12VideoReader(str(video_path), width=width, height=height, device="cpu") as reader:
        (y_tensor, u_tensor, v_tensor), success = reader.read_frame()

    assert success is True
    assert tuple(y_tensor.shape) == (1, 1, 4, 4)
    assert tuple(u_tensor.shape) == (1, 1, 2, 2)
    assert tuple(v_tensor.shape) == (1, 1, 2, 2)
    np.testing.assert_array_equal(u_tensor.squeeze().numpy(), np.array([[100, 101], [102, 103]]))
    np.testing.assert_array_equal(v_tensor.squeeze().numpy(), np.array([[150, 151], [152, 153]]))


def test_async_yuv_writer_and_save_yuv_write_expected_bytes(tmp_path):
    frame_a = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    frame_b = torch.tensor([5, 6, 7, 8], dtype=torch.uint8)

    sync_path = tmp_path / "sync.yuv"
    save_yuv(frame_a, str(sync_path), append=False)
    save_yuv(frame_b, str(sync_path), append=True)
    assert sync_path.read_bytes() == bytes([1, 2, 3, 4, 5, 6, 7, 8])

    async_path = tmp_path / "async.yuv"
    with AsyncYUVWriter(str(async_path)) as writer:
        writer.write(frame_a)
        writer.write(frame_b)

    assert async_path.read_bytes() == bytes([1, 2, 3, 4, 5, 6, 7, 8])
