"""Benchmark fp32 eager vs the INT8+FP16 mixed-precision TensorRT engine.

Compares wall-clock latency and memory footprint on the same image + prompt:
  - fp32 eager (PyTorch, detector_model) — runs on CPU on this box, since
    torch.cuda is unavailable here (driver too old, see README).
  - INT8+FP16 mixed-precision TensorRT engine (artifacts/sam3.engine) — runs
    on GPU.

These measure different resources (CPU RSS vs GPU memory) because that's
where each model actually runs on this machine — not an apples-to-apples
memory comparison, just what each backend costs on the hardware it uses.

Run after build_engine.py (needs artifacts/sam3.engine to exist).

pip install tensorrt pycuda pillow numpy
"""
import platform
import resource
import statistics
import time

import numpy as np
import torch

from constant import DEVICE, ENGINE, IMAGE_PATH, PROMPT
from model_utils import load_tokenizer, load_wrapped_detector, preprocess_image, tokenize_prompt

N_BENCHMARK_RUNS = 10


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB (ru_maxrss unit differs by OS)."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / 1024 if platform.system() == "Linux" else ru_maxrss / (1024 * 1024)


def summarize(times: list[float]) -> str:
    ms = [t * 1000 for t in times]
    mean = statistics.mean(ms)
    stdev = statistics.stdev(ms) if len(ms) > 1 else 0.0
    return f"{mean:.1f} ms +/- {stdev:.1f} ms (min {min(ms):.1f}, max {max(ms):.1f})"


def benchmark_fp32_eager(wrapped, pixel_values, input_ids) -> tuple[list[float], float]:
    with torch.no_grad():
        wrapped(pixel_values, input_ids)  # warm-up, not timed
    times = []
    for _ in range(N_BENCHMARK_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            wrapped(pixel_values, input_ids)
        times.append(time.perf_counter() - t0)
    return times, peak_rss_mb()


def benchmark_trt_engine(image_np: np.ndarray, tokenized_text_np: np.ndarray) -> tuple[list[float], float]:
    import pycuda.autoinit  # noqa: F401  (creates a CUDA context via the driver API)
    import pycuda.driver as cuda

    from infer import TrtRunner  # reuses the pycuda-backed engine wrapper from infer.py

    runner = TrtRunner(ENGINE)
    feeds = {"image": image_np, "tokenized_text": tokenized_text_np}
    runner.infer(feeds)  # warm-up, not timed
    times = []
    for _ in range(N_BENCHMARK_RUNS):
        t0 = time.perf_counter()
        runner.infer(feeds)
        times.append(time.perf_counter() - t0)
    free_bytes, total_bytes = cuda.mem_get_info()
    used_mb = (total_bytes - free_bytes) / (1024 * 1024)
    return times, used_mb


def main() -> None:
    if not ENGINE.exists():
        raise FileNotFoundError(f"{ENGINE} not found — run build_engine.py first")

    tokenizer = load_tokenizer()
    pixel_values = preprocess_image(IMAGE_PATH, DEVICE)
    input_ids = tokenize_prompt(tokenizer, PROMPT, DEVICE)

    print(f"fp32 eager ({N_BENCHMARK_RUNS} runs, CPU)...")
    wrapped = load_wrapped_detector(DEVICE, dtype=torch.float32)
    fp32_times, fp32_rss_mb = benchmark_fp32_eager(wrapped, pixel_values, input_ids)
    del wrapped

    print(f"INT8+FP16 TensorRT engine ({N_BENCHMARK_RUNS} runs, GPU)...")
    trt_times, trt_gpu_mb = benchmark_trt_engine(
        pixel_values.cpu().numpy(), input_ids.cpu().numpy()
    )

    print()
    print(f"  fp32 eager (CPU):          {summarize(fp32_times)}   peak process RSS: {fp32_rss_mb:.0f} MB")
    print(f"  INT8+FP16 TensorRT (GPU):  {summarize(trt_times)}   GPU memory in use: {trt_gpu_mb:.0f} MB")
    speedup = statistics.mean(fp32_times) / statistics.mean(trt_times)
    print(f"  speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
