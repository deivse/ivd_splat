from pathlib import Path
from typing import NamedTuple, Optional

import torch


def human_readable_memory_size(size: int) -> str:
    """
    Convert memory size in bytes
    to human readable format.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


class CudaMemStats(NamedTuple):
    device: str
    allocated: int
    cached: int

    def log_str(self, message: Optional[str] = None, separator="\n") -> str:
        output = ""
        output += f"Cuda stats ({message or ''} {self.device}):{separator}"
        output += (
            f"Memory Allocated: {human_readable_memory_size(self.allocated)}{separator}"
        )
        output += f"Memory Cached: {human_readable_memory_size(self.cached)}{separator}"
        return output

    def allocated_str(self) -> str:
        return human_readable_memory_size(self.allocated)

    def cached_str(self) -> str:
        return human_readable_memory_size(self.cached)

    def __str__(self) -> str:
        return self.log_str(separator=" ")

    @classmethod
    def for_device(cls, device: str) -> "CudaMemStats":
        device_index = int(device.removeprefix("cuda:"))
        return cls(
            device=device,
            allocated=torch.cuda.memory_allocated(device_index),
            cached=torch.cuda.memory_reserved(device_index),
        )


def cuda_stats_msg(device: str, message: Optional[str] = None) -> str:
    if not device.startswith("cuda"):
        return "Not a CUDA device"
    return CudaMemStats.for_device(device).log_str(message)


class CudaMemSnapshotter:
    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        torch.cuda.memory._record_memory_history(enabled="state")
        print(f"Initialized CudaMemSnapshotter, will dump snapshot to {snapshot_path} on deletion.")

    def __del__(self):
        torch.cuda.memory._dump_snapshot(str(self.snapshot_path))
        print(f"Dumped CUDA memory snapshot to {self.snapshot_path}")
