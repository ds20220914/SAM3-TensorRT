"""Build a TensorRT INT8+FP16 engine from the QDQ ONNX graph produced by export_onnx.py.

Run on the GPU box, after quantize.py and export_onnx.py. TensorRT reads the
QuantizeLinear/DequantizeLinear nodes in the ONNX graph directly (explicit
quantization) — no separate INT8 calibrator is needed. INT8+FP16 hybrid
precision: TensorRT picks per-layer, so anything it can't/shouldn't run in
pure INT8 falls back to FP16 instead of failing the build.

pip install tensorrt
"""
import os
import time

import tensorrt as trt

from constant import BUILDER_OPTIMIZATION_LEVEL, ENGINE, QDQ_ONNX, TIMING_CACHE

# Override via env vars for memory-constrained boards (e.g. Jetson Orin
# Nano's 8GB shared CPU/GPU pool, where 4 GiB just for the builder's tactic
# scratch space starves everything else) — e.g.:
#   TRT_WORKSPACE_MB=512 TRT_OPT_LEVEL=2 python3 build_engine.py
WORKSPACE_BYTES = int(os.environ.get("TRT_WORKSPACE_MB", 4096)) << 20
OPT_LEVEL = int(os.environ.get("TRT_OPT_LEVEL", BUILDER_OPTIMIZATION_LEVEL))


def build_int8_engine() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {QDQ_ONNX.name}")
    if not parser.parse(QDQ_ONNX.read_bytes(), path=str(QDQ_ONNX)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {QDQ_ONNX}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.builder_optimization_level = OPT_LEVEL
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    cache_bytes = TIMING_CACHE.read_bytes() if TIMING_CACHE.exists() else b""
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    print(
        f"Building INT8+FP16 engine (opt-level {OPT_LEVEL}, "
        f"workspace {WORKSPACE_BYTES / (1 << 20):.0f} MB). "
        "First build with no timing cache can take 15-30 min."
    )
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")
    dt = time.perf_counter() - t0

    ENGINE.write_bytes(serialized_engine)
    print(f"  built in {dt:.0f}s -> {ENGINE} ({ENGINE.stat().st_size / 1e9:.2f} GB)")

    TIMING_CACHE.write_bytes(bytes(config.get_timing_cache().serialize()))
    print(f"  {TIMING_CACHE}")


if __name__ == "__main__":
    build_int8_engine()
