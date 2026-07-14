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

from constant import CONTEXT_LENGTH, IMAGE_SIZE, MEAN, MODEL_PATH, STD


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


def load_wrapped_detector(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> Sam3DetectorWrapper:
    """Load the local checkpoint and return the wrapped detector, patched for IMAGE_SIZE."""
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
    return Sam3DetectorWrapper(model.detector_model).eval().to(device)


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
