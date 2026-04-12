import numpy as np
import torch


class RAWVideoReader:
    """
    Reader for RAW video files.
    """

    def __init__(self, video_path: str, config: dict, device: str = "cuda"):
        """
        Args:
            video_path: Path to the RAW video file
            config: Camera configuration dictionary
            device: Target device for loaded frames
        """
        self.video_path = video_path
        self.config = config

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        self.width = config["img"]["width"]
        self.height = config["img"]["height"]
        self.emb_lines = config["img"]["emb_lines"]

        self.pixels_per_frame = self.width * self.height
        self.bytes_per_pixel = 2
        self.frame_size_bytes = self.pixels_per_frame * self.bytes_per_pixel

        top_lines, bottom_lines = self.emb_lines
        self.output_height = self.height - top_lines - bottom_lines
        self.output_width = self.width

        self.file_handle = None

    def __enter__(self):
        self.file_handle = open(self.video_path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle is not None:
            self.file_handle.close()

    def read_frame(self) -> tuple[torch.Tensor, bool]:
        """
        Read one frame from the video.

        Returns:
            tuple: (frame_tensor, success)
                - frame_tensor: [H, W] uint16 tensor on self.device
                - success: True if a frame was read, False on end of file
        """
        if self.file_handle is None:
            raise RuntimeError("RAWVideoReader must be used as context manager")

        frame_bytes = self.file_handle.read(self.frame_size_bytes)

        if len(frame_bytes) < self.frame_size_bytes:
            return None, False

        frame_np = np.frombuffer(frame_bytes, dtype=np.uint16)

        frame_1d = torch.from_numpy(frame_np.copy())

        frame_2d = frame_1d.reshape(self.height, self.width)

        top_lines, bottom_lines = self.emb_lines
        if top_lines > 0:
            frame_2d = frame_2d[top_lines:, :]
        if bottom_lines > 0:
            frame_2d = frame_2d[:-bottom_lines, :]

        frame_2d = frame_2d.to(self.device)

        return frame_2d, True

    def __iter__(self):
        if self.file_handle is None:
            raise RuntimeError("RAWVideoReader must be used as context manager")

        self.file_handle.seek(0)
        frame_number = 0

        while True:
            frame, success = self.read_frame()
            if not success:
                break

            frame_number += 1
            yield frame, frame_number


def read_video_frames_generator(video_path: str, config: dict, device: str = "cuda"):
    """
    Generator for frame-by-frame RAW video reading.

    Args:
        video_path: Path to the binary video file
        config: Camera configuration dictionary
        device: Target device for loaded frames

    Yields:
        tuple: (frame_tensor, frame_number)
    """
    with RAWVideoReader(video_path, config, device) as reader:
        yield from reader
