# src/kmer_ord/utils/benchmark.py
import psutil
import csv
import os
import time
import threading
from datetime import datetime


def _tree_rss(process: "psutil.Process") -> int:
    """Resident set size of a process plus all its children (bytes).

    Counting stages and DR spawn subprocesses, so the parent's own RSS would
    understate true memory use; we sum the whole process tree.
    """
    total = 0
    try:
        total += process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


class BenchmarkTimer:
    """
    Context manager for timing, CPU, and memory usage of operations.
    Automatically logs metrics to TSV.

    Peak memory is sampled by a background thread (including child processes),
    because the previous end-minus-start delta could not observe a spike that
    occurs and is freed within the block - exactly the pattern that caused OOMs.
    """
    def __init__(self, label="Run", log_dir="benchmarking", script_name=None,
                 input_file=None, input_args=None, sample_interval=0.05):
        self.label = label
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "benchmark_log.tsv")
        self.script_name = script_name
        self.input_file = str(input_file) if input_file else None
        self.input_args = input_args
        self.sample_interval = sample_interval

        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = None
        self.start_cpu_time = None
        self.start_memory = None
        self.peak_memory = 0

        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._sampler_thread = None

    def _sample_peak(self):
        while not self._stop_event.is_set():
            self.peak_memory = max(self.peak_memory, _tree_rss(self._process))
            self._stop_event.wait(self.sample_interval)

    def __enter__(self):
        self.start_time = time.time()
        self.start_cpu_time = time.process_time()
        self.start_memory = self._process.memory_info().rss
        self.peak_memory = _tree_rss(self._process)

        self._stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sample_peak, daemon=True)
        self._sampler_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join()

        end_time = time.time()
        end_cpu_time = time.process_time()
        end_memory = self._process.memory_info().rss
        self.peak_memory = max(self.peak_memory, _tree_rss(self._process))

        self.wall_time = end_time - self.start_time
        self.cpu_time = end_cpu_time - self.start_cpu_time
        self.memory_used = max(0, end_memory - self.start_memory)

        self._log_metrics()
    
    def _log_metrics(self):
        log_exists = os.path.isfile(self.log_file)
        input_file_size = os.path.getsize(self.input_file) if self.input_file and os.path.exists(self.input_file) else "N/A"

        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter='\t')
            if not log_exists:
                writer.writerow([
                    "timestamp", "script_name", "input_file", "input_file_size_bytes",
                    "input_args", "label", "wall_time_s", "cpu_time_s",
                    "memory_used_bytes", "peak_memory_bytes"
                ])
            writer.writerow([
                datetime.now().isoformat(),
                self.script_name or "N/A",
                self.input_file or "N/A",
                input_file_size,
                self.input_args or "N/A",
                self.label,
                self.wall_time,
                self.cpu_time,
                self.memory_used,
                self.peak_memory
            ])
