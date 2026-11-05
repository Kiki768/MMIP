"""
GPU 顯存（VRAM）監測工具。

同時記錄兩種角度的數字，報告中要說清楚差別：
  1. PyTorch allocator（torch.cuda.memory_allocated / max_memory_allocated）
     → 張量實際占用的記憶體，不含 CUDA context 與快取碎片，最適合比較「模型本身」的大小。
  2. NVML（nvidia-smi 看到的數字）
     → 整張卡已使用量，含 CUDA context（約 300~500 MiB）、cache allocator 保留區與其他行程；
        以背景執行緒每 interval 秒取樣一次，求平均與峰值。

用法：
    with GPUMemoryMonitor() as mon:
        model.generate(...)
    print(mon.stats())
"""
import threading
import time

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

MiB = 1024 ** 2


class GPUMemoryMonitor:
    def __init__(self, device=0, interval=0.05):
        self.device = device
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device) if _NVML_OK else None

    def _poll(self):
        while not self._stop.is_set():
            self._samples.append(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / MiB)
            time.sleep(self.interval)

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_alloc = torch.cuda.memory_allocated(self.device) / MiB
        self._t0 = time.perf_counter()
        if self._handle is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._peak_alloc = torch.cuda.max_memory_allocated(self.device) / MiB
        self._peak_reserved = torch.cuda.max_memory_reserved(self.device) / MiB
        return False

    def stats(self):
        s = {
            "torch_start_alloc_mib": round(self._start_alloc, 1),
            "torch_peak_alloc_mib": round(self._peak_alloc, 1),
            "torch_peak_reserved_mib": round(self._peak_reserved, 1),
            "elapsed_s": round(self.elapsed, 3),
        }
        if self._samples:
            s["nvml_avg_used_mib"] = round(sum(self._samples) / len(self._samples), 1)
            s["nvml_peak_used_mib"] = round(max(self._samples), 1)
            s["nvml_num_samples"] = len(self._samples)
        return s


def free_gpu():
    """切換精度前先 `del model` 再呼叫本函式清空快取，避免上一個模型殘留影響量測。"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
