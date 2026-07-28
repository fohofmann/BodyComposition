"""Logging adapters for pipeline output and memory diagnostics."""

import logging

from psutil import virtual_memory


class LoggingWriter:
    """Redirect writes from a file-like stream to the Python logger."""

    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logging.log(self.level, message.rstrip())

    def flush(self):
        pass


def log_gpu_usage(device=None):
    """Log process RAM and, when available, peak CUDA allocation."""

    memory_info = virtual_memory()
    msg = " usage:"
    msg += (
        f" RAM {round(memory_info.used / (1024**3), 1)}/{round(memory_info.total / (1024**3), 1)}GB"
    )
    try:
        import torch.cuda as cuda
    except ImportError:
        cuda = None
    if cuda is not None and cuda.is_available():
        msg += (
            f" | VRAM {round(cuda.max_memory_allocated(device) / (1024**3), 1)}/"
            f"{round(cuda.max_memory_reserved(device) / (1024**3), 1)}GB"
        )
    logging.info(msg)
