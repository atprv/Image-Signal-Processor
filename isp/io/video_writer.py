from queue import Empty, Queue
from threading import Thread

import torch


class AsyncYUVWriter:
    """
    Async writer for YUV files.
    """

    def __init__(self, output_path: str, queue_size: int = 10):
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
        self.worker_error = None

    def _raise_worker_error(self):
        """Re-raise exceptions captured by the writer thread."""
        if self.worker_error is None:
            return

        error = self.worker_error
        self.worker_error = None
        raise RuntimeError("Async YUV writer failed.") from error

    def _writer_worker(self):
        """Worker thread for frame writing."""
        try:
            with open(self.output_path, "wb") as file_handle:
                self.file_handle = file_handle

                while self.is_running or not self.queue.empty():
                    try:
                        yuv_frame = self.queue.get(timeout=0.1)
                    except Empty:
                        continue

                    if yuv_frame is None:
                        self.queue.task_done()
                        break

                    if yuv_frame.is_cuda:
                        yuv_frame = yuv_frame.cpu()

                    file_handle.write(yuv_frame.numpy().tobytes())
                    self.queue.task_done()
        except Exception as error:
            self.worker_error = error
        finally:
            self.file_handle = None

    def start(self):
        """Start the writer thread."""
        if self.is_running:
            return

        self.worker_error = None
        self.is_running = True
        self.writer_thread = Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def write(self, yuv_frame: torch.Tensor):
        """
        Add a frame to the write queue.

        Args:
            yuv_frame: NV12 YUV frame as a uint8 tensor
        """
        self._raise_worker_error()

        if not self.is_running:
            raise RuntimeError("Writer not started. Call start() first.")

        self.queue.put(yuv_frame)

    def finish(self):
        """Finish writing and wait for the worker thread."""
        if not self.is_running:
            self._raise_worker_error()
            return

        if self.writer_thread is not None and self.writer_thread.is_alive():
            self.queue.put(None)

        self.is_running = False
        if self.writer_thread is not None:
            self.writer_thread.join()
            self.writer_thread = None

        self._raise_worker_error()

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

    with open(filename, mode) as file_handle:
        if yuv_frame.is_cuda:
            yuv_frame = yuv_frame.cpu()

        file_handle.write(yuv_frame.numpy().tobytes())
