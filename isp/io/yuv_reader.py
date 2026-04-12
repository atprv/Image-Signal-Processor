import numpy as np
import torch


class NV12VideoReader:
    """
    Reader for NV12 YUV420 video (Y plane + interleaved UV).
    """

    def __init__(self, video_path: str, width: int, height: int, device: str = "cuda"):
        """
        Args:
            video_path: Path to the NV12 YUV file
            width: Frame width
            height: Frame height
            device: Target device for loaded frames
        """
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(f"NV12 requires even width and height, got {width}x{height}")

        self.video_path = video_path
        self.width = width
        self.height = height

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)

        self.y_size = width * height
        self.uv_width = width // 2
        self.uv_height = height // 2
        self.uv_size = 2 * self.uv_width * self.uv_height
        self.frame_size_bytes = self.y_size + self.uv_size

        self.file_handle = None

    def __enter__(self):
        self.file_handle = open(self.video_path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle is not None:
            self.file_handle.close()

    def read_frame(self) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, bool]:
        """
        Read one frame from the NV12 video.

        Returns:
            tuple: ((y, u, v), success)
                - y: [1, 1, H, W] uint8 tensor
                - u: [1, 1, H/2, W/2] uint8 tensor
                - v: [1, 1, H/2, W/2] uint8 tensor
                - success: True if a frame was read, False on end of file
        """
        if self.file_handle is None:
            raise RuntimeError("NV12VideoReader must be used as context manager")

        frame_bytes = self.file_handle.read(self.frame_size_bytes)
        if len(frame_bytes) < self.frame_size_bytes:
            return None, False

        y_np = np.frombuffer(frame_bytes[: self.y_size], dtype=np.uint8)
        y = torch.from_numpy(y_np.copy()).reshape(self.height, self.width)
        y = y.unsqueeze(0).unsqueeze(0)

        uv_np = np.frombuffer(frame_bytes[self.y_size :], dtype=np.uint8)
        uv = uv_np.reshape(self.uv_height, self.uv_width, 2)

        u = torch.from_numpy(uv[:, :, 0].copy()).unsqueeze(0).unsqueeze(0)
        v = torch.from_numpy(uv[:, :, 1].copy()).unsqueeze(0).unsqueeze(0)

        return (y.to(self.device), u.to(self.device), v.to(self.device)), True

    def __iter__(self):
        if self.file_handle is None:
            raise RuntimeError("NV12VideoReader must be used as context manager")

        self.file_handle.seek(0)
        frame_number = 0

        while True:
            frame, success = self.read_frame()
            if not success:
                break

            frame_number += 1
            yield frame, frame_number


def read_video_frames_generator(video_path: str, width: int, height: int, device: str = "cuda"):
    """
    Generator for frame-by-frame NV12 video reading.

    Args:
        video_path: Path to the YUV file
        width: Frame width
        height: Frame height
        device: Target device for loaded frames

    Yields:
        tuple: ((y, u, v), frame_number)
    """
    with NV12VideoReader(video_path, width, height, device=device) as reader:
        yield from reader
