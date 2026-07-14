"""Shared paths and configuration for the SAM3 image-only INT8/TensorRT pipeline."""
from pathlib import Path

import torch

ARTIFACTS_PATH = Path("artifacts")
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "./sam3"  # local checkpoint (Sam3VideoModel weights; we only use .detector_model)

IMAGE_SIZE = 924  # multiple of patch size (14); embedl-deploy tutorial's tested config
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
