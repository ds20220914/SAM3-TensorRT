"""Shared model/pre-processing helpers for the SAM3 image (PCS) INT8/TensorRT pipeline.

We only ever use the *detector* half of the checkpoint (Sam3VideoModel.detector_model),
since Promptable Concept Segmentation on a single image is exactly what that submodule
does — the tracker/memory-attention half of the checkpoint is video-only and unused here.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Sam3VideoModel
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from constant import CONTEXT_LENGTH, IMAGE_SIZE, MEAN, MODEL_PATH, STD

# Sam3VisionModel returns this many FPN levels (verified empirically at
# IMAGE_SIZE=924: 4 levels, [264,132,66,33] per side) — a structural fact of
# the vision backbone config, not a tunable. Sam3FusionWrapper needs it to
# know where to split its flattened tuple input back into the two lists
# (fpn_hidden_states, fpn_position_encoding) that Sam3VisionEncoderOutput
# expects.
N_FPN_LEVELS = 4

# Shared between export_onnx_split.py (ONNX input/output names) and
# infer_split.py (TensorRT tensor names) so the vision engine's outputs and
# the fusion engine's inputs line up by name, not by fragile positional order.
FPN_TENSOR_NAMES = [f"fpn_hidden_{i}" for i in range(N_FPN_LEVELS)] + [
    f"fpn_pos_{i}" for i in range(N_FPN_LEVELS)
]


def patch_image_size(cfg: AutoConfig, image_size: int) -> AutoConfig:
    """Rewrite the checkpoint config so the vision backbone matches ``image_size``."""
    grid = image_size // 14
    feat_sizes = [[grid * 4, grid * 4], [grid * 2, grid * 2], [grid, grid]]
    cfg.image_size = image_size
    cfg.low_res_mask_size = grid * 4
    for sub in (cfg.detector_config, cfg.tracker_config):
        sub.image_size = image_size
        sub.vision_config.backbone_feature_sizes = feat_sizes
        sub.vision_config.backbone_config.image_size = image_size
    cfg.tracker_config.memory_attention_rope_feat_sizes = [grid, grid]
    return cfg


class Sam3DetectorWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper around the SAM3 detector, for export/TRT."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.detector(pixel_values=pixel_values, input_ids=input_ids)
        return out.pred_masks, out.pred_logits, out.pred_boxes


class Sam3VisionEncoderWrapper(nn.Module):
    """Vision backbone only — pixel_values -> flattened multi-scale FPN features.

    Split out of Sam3DetectorWrapper so it can be exported/quantized/built as
    its own TensorRT engine, separate from Sam3FusionWrapper below. This is
    the expensive half (a ViT-style backbone over a 924x924 image) — splitting
    it off is what actually reduces peak memory during TensorRT engine
    building on memory-constrained boards (e.g. Jetson Orin Nano), since the
    builder no longer has to hold the whole fused graph's tactic-search
    buffers in memory at once.
    """

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = detector.vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.vision_encoder(pixel_values)
        return (*out.fpn_hidden_states, *out.fpn_position_encoding)


class Sam3FusionWrapper(nn.Module):
    """Text encoding + detection/mask head — takes precomputed vision FPN
    features (Sam3VisionEncoderWrapper's output) instead of pixel_values.

    Reuses the checkpoint's own vision_embeds= re-entry point (verified to
    produce bit-identical output vs. the single fused pixel_values path —
    see conversation/JETSON_MIGRATION.md) rather than re-implementing the
    detector's internal fusion logic by hand.
    """

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(
        self, *fpn_and_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = N_FPN_LEVELS
        fpn_hidden_states = list(fpn_and_ids[:n])
        fpn_position_encoding = list(fpn_and_ids[n : 2 * n])
        input_ids = fpn_and_ids[2 * n]
        vision_embeds = Sam3VisionEncoderOutput(
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )
        out = self.detector(vision_embeds=vision_embeds, input_ids=input_ids)
        return out.pred_masks, out.pred_logits, out.pred_boxes


def _load_detector(device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load the local checkpoint's detector submodule, patched for IMAGE_SIZE."""
    cfg = patch_image_size(
        AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True), IMAGE_SIZE
    )
    model = (
        Sam3VideoModel.from_pretrained(
            MODEL_PATH,
            config=cfg,
            torch_dtype=dtype,
            local_files_only=True,
            # sdpa's shape-dispatch fast path compares an unbacked symint
            # (spatial_shapes pulled from a tensor via .item() inside
            # _get_rpb_matrix) against a concrete shape, which torch.export
            # can't resolve (GuardOnDataDependentSymNode). Eager attention
            # skips that dispatch check entirely.
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )
    return model.detector_model


def load_wrapped_detector(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> Sam3DetectorWrapper:
    """Load the checkpoint and return the single-engine (unsplit) wrapped detector."""
    return Sam3DetectorWrapper(_load_detector(device, dtype)).eval().to(device)


def load_vision_encoder(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> Sam3VisionEncoderWrapper:
    """Load the checkpoint and return just the vision-encoder half (engine A)."""
    return Sam3VisionEncoderWrapper(_load_detector(device, dtype)).eval().to(device)


def load_fusion_head(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> Sam3FusionWrapper:
    """Load the checkpoint and return the text+fusion half (engine B)."""
    return Sam3FusionWrapper(_load_detector(device, dtype)).eval().to(device)


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)


def tokenize_prompt(
    tokenizer: AutoTokenizer, prompt: str, device: torch.device
) -> torch.Tensor:
    """Tokenize a text prompt to a fixed-length (CONTEXT_LENGTH) input_ids tensor."""
    ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",
    ).input_ids
    return ids.to(device)


def preprocess_image(path: Path, device: torch.device) -> torch.Tensor:
    """Load a JPEG, resize to IMAGE_SIZE, and normalize to match SAM3 training preprocessing."""
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0).to(device)
