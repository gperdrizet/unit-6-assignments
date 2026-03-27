"""
Background thread that polls system RAM and GPU VRAM usage at a fixed interval.

Usage:
    monitor = MemoryMonitor(cuda_device_index=0)
    monitor.start()
    # ... run work ...
    monitor.stop()
    print(monitor.peak_system_mb, monitor.peak_gpu_mb)
"""

import threading
import time

import psutil
import torch


class MemoryMonitor:
    """Polls system RAM and GPU VRAM on a background thread.

    Parameters
    ----------
    cuda_device_index : int
        The CUDA device index (as seen by PyTorch after CUDA_VISIBLE_DEVICES
        remapping) to monitor.  Usually 0 when CUDA_VISIBLE_DEVICES is set to
        a single device.
    interval : float
        Polling interval in seconds (default 0.5).
    """

    def __init__(self, cuda_device_index: int = 0, interval: float = 0.5):
        self._device_index = cuda_device_index
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._process = psutil.Process()

        # Filled in by start() and updated each poll tick
        self.baseline_system_mb: float = 0.0
        self.peak_system_mb: float = 0.0
        self.peak_gpu_mb: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Snapshot baseline RSS, reset VRAM tracking, then begin polling."""
        self._stop_event.clear()

        # Baseline – before the work starts
        self.baseline_system_mb = self._current_system_mb()

        # Reset PyTorch VRAM peak counter for this device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device_index)

        self.peak_system_mb = self.baseline_system_mb
        self.peak_gpu_mb = 0.0

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 4)
        # Final snapshot after work has finished
        self._update_peaks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._update_peaks()
            self._stop_event.wait(timeout=self._interval)

    def _update_peaks(self) -> None:
        current_sys = self._current_system_mb()
        if current_sys > self.peak_system_mb:
            self.peak_system_mb = current_sys

        if torch.cuda.is_available():
            # max_memory_allocated returns bytes; peak is tracked by PyTorch
            gpu_bytes = torch.cuda.max_memory_allocated(self._device_index)
            gpu_mb = gpu_bytes / (1024 ** 2)
            if gpu_mb > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu_mb

    def _current_system_mb(self) -> float:
        return self._process.memory_info().rss / (1024 ** 2)
