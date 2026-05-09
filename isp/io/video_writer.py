from queue import Queue
from threading import Thread

import torch


class AsyncYUVWriter:
    """
    Async writer for YUV files.
    """

    def __init__(self, output_path: str, queue_size: int = 32):
        """
        Args:
            output_path: Path to the output YUV file
            queue_size: Frame queue size
        """
        self.output_path = output_path
        self.queue = Queue(maxsize=queue_size)
        self.writer_thread = None
        self.is_running = False
        self.file_handle = None

    def _writer_worker(self):
        """Worker thread for frame writing."""
        with open(self.output_path, "wb") as f:
            self.file_handle = f

            while self.is_running or not self.queue.empty():
                try:
                    queue_item = self.queue.get(timeout=0.1)

                    if queue_item is None:
                        break

                    copy_done_event = None
                    yuv_frame = queue_item
                    if isinstance(queue_item, tuple):
                        yuv_frame, copy_done_event = queue_item

                    if copy_done_event is not None:
                        copy_done_event.synchronize()

                    if yuv_frame.is_cuda:
                        yuv_frame = yuv_frame.cpu()

                    yuv_np = yuv_frame.numpy()
                    f.write(memoryview(yuv_np))

                    self.queue.task_done()

                except Exception:
                    continue

        self.file_handle = None

    def start(self):
        """Start the writer thread."""
        if self.is_running:
            return

        self.is_running = True
        self.writer_thread = Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def write(self, yuv_frame):
        """
        Add a frame to the write queue.

        Args:
            yuv_frame: NV12 YUV frame as a uint8 tensor, or
                ``(tensor, cuda_event)`` for async GPU->CPU copies.
        """
        if not self.is_running:
            raise RuntimeError("Writer not started. Call start() first.")

        self.queue.put(yuv_frame)

    def finish(self):
        """Finish writing and wait for the worker thread."""
        if not self.is_running:
            return

        self.queue.put(None)

        self.is_running = False
        if self.writer_thread is not None:
            self.writer_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def save_yuv(yuv_frame: torch.Tensor, filename: str, append: bool = True):
    """
    Synchronous helper to save a YUV frame.

    Args:
        yuv_frame: YUV tensor (uint8, 1D, size H*W*3/2)
        filename: Output filename
        append: If True, append the frame to the file
    """
    mode = "ab" if append else "wb"

    with open(filename, mode) as f:
        if yuv_frame.is_cuda:
            yuv_frame = yuv_frame.cpu()

        yuv_np = yuv_frame.numpy()
        f.write(memoryview(yuv_np))
