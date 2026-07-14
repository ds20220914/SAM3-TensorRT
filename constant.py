"""Shared paths and configuration for the SAM3 image-only INT8/TensorRT pipeline."""
from pathlib import Path

import torch

ARTIFACTS_PATH = Path("artifacts")
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "./sam3"  # local checkpoint (Sam3VideoModel weights; we only use .detector_model)

IMAGE_SIZE = 924  # multiple of patch size (14)
PATCH_SIZE = 14
CONTEXT_LENGTH = 32  # CLIP text tower max_position_embeddings / tokenizer model_max_length
PROMPT = "white bag"
N_CALIB = 100  # frames used for PTQ calibration
BUILDER_OPTIMIZATION_LEVEL = 3

CONFIDENCE = 0.4  # infer.py: minimum sigmoid(pred_logits) score to keep a query
NMS_IOU_THRESHOLD = 0.5  # infer.py: dedupe overlapping queries firing on the same bag

CALIB_PATH = Path("calibration_images")
IMAGE_PATH = Path("images/BAG.png")

# Normalization — must match the SAM3 training-time preprocessing.
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

FP32_PT2 = ARTIFACTS_PATH / f"sam3_resized_{IMAGE_SIZE}.pt2"
QDQ_PT2 = ARTIFACTS_PATH / f"sam3_resized_{IMAGE_SIZE}_int8_qdq.pt2"
QDQ_ONNX = ARTIFACTS_PATH / f"sam3_resized_{IMAGE_SIZE}_int8_qdq.onnx"
ENGINE = ARTIFACTS_PATH / "sam3.engine"
TIMING_CACHE = ARTIFACTS_PATH / "trt_timing.cache"

# Two-engine split (vision encoder / text+fusion head), for memory-constrained
# boards (e.g. Jetson Orin Nano) where building the single fused engine above
# needs more TensorRT builder workspace than the board has RAM for — see
# JETSON_MIGRATION.md. Same artifacts, just built/run as two smaller pieces
# instead of one big one; torch_export_split.py/quantize_split.py/
# export_onnx_split.py/build_engine_split.py/infer_split.py are the
# split-pipeline equivalents of the single-engine scripts above.
VISION_FP32_PT2 = ARTIFACTS_PATH / f"sam3_vision_{IMAGE_SIZE}.pt2"
VISION_QDQ_PT2 = ARTIFACTS_PATH / f"sam3_vision_{IMAGE_SIZE}_int8_qdq.pt2"
VISION_QDQ_ONNX = ARTIFACTS_PATH / f"sam3_vision_{IMAGE_SIZE}_int8_qdq.onnx"
VISION_ENGINE = ARTIFACTS_PATH / "sam3_vision.engine"
VISION_TIMING_CACHE = ARTIFACTS_PATH / "trt_timing_vision.cache"

FUSION_FP32_PT2 = ARTIFACTS_PATH / f"sam3_fusion_{IMAGE_SIZE}.pt2"
FUSION_QDQ_PT2 = ARTIFACTS_PATH / f"sam3_fusion_{IMAGE_SIZE}_int8_qdq.pt2"
FUSION_QDQ_ONNX = ARTIFACTS_PATH / f"sam3_fusion_{IMAGE_SIZE}_int8_qdq.onnx"
FUSION_ENGINE = ARTIFACTS_PATH / "sam3_fusion.engine"
FUSION_TIMING_CACHE = ARTIFACTS_PATH / "trt_timing_fusion.cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
