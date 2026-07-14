"""Build two TensorRT INT8+FP16 engines (vision encoder, text+fusion head)
from the QDQ ONNX graphs produced by export_onnx_split.py.

Split-pipeline equivalent of build_engine.py — see JETSON_MIGRATION.md.
Building each half separately keeps peak TensorRT-builder memory well below
what building the single fused engine needs, which is the whole point on
memory-constrained boards (e.g. Jetson Orin Nano's 8GB shared CPU/GPU pool).

Same TRT_WORKSPACE_MB / TRT_OPT_LEVEL env var overrides as build_engine.py:
  TRT_WORKSPACE_MB=512 TRT_OPT_LEVEL=2 python3 build_engine_split.py
"""
import os
import time

import tensorrt as trt

from constant import (
    BUILDER_OPTIMIZATION_LEVEL,
    FUSION_ENGINE,
    FUSION_QDQ_ONNX,
    FUSION_TIMING_CACHE,
    VISION_ENGINE,
    VISION_QDQ_ONNX,
    VISION_TIMING_CACHE,
)

WORKSPACE_BYTES = int(os.environ.get("TRT_WORKSPACE_MB", 4096)) << 20
OPT_LEVEL = int(os.environ.get("TRT_OPT_LEVEL", BUILDER_OPTIMIZATION_LEVEL))


def build_engine(onnx_path, engine_path, timing_cache_path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {onnx_path.name}")
    if not parser.parse(onnx_path.read_bytes(), path=str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.builder_optimization_level = OPT_LEVEL
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    print(
        f"Building {onnx_path.stem} (opt-level {OPT_LEVEL}, "
        f"workspace {WORKSPACE_BYTES / (1 << 20):.0f} MB)..."
    )
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    dt = time.perf_counter() - t0

    engine_path.write_bytes(serialized_engine)
    print(f"  built in {dt:.0f}s -> {engine_path} ({engine_path.stat().st_size / 1e9:.2f} GB)")

    timing_cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    print(f"  {timing_cache_path}")


if __name__ == "__main__":
    build_engine(VISION_QDQ_ONNX, VISION_ENGINE, VISION_TIMING_CACHE)
    build_engine(FUSION_QDQ_ONNX, FUSION_ENGINE, FUSION_TIMING_CACHE)
