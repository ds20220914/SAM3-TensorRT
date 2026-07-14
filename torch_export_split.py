"""Export the SAM3 detector as two fp32 .pt2 files instead of one: a vision-
encoder engine and a text+fusion-head engine. Split-pipeline equivalent of
torch_export.py — see JETSON_MIGRATION.md for why (TensorRT builder memory on
boards like Jetson Orin Nano).

Run this first, on the GPU box. Produces:
  artifacts/sam3_vision_<IMAGE_SIZE>.pt2
  artifacts/sam3_fusion_<IMAGE_SIZE>.pt2
consumed by quantize_split.py.
"""
import torch

from constant import (
    CONTEXT_LENGTH,
    DEVICE,
    FUSION_FP32_PT2,
    IMAGE_SIZE,
    VISION_FP32_PT2,
)
from model_utils import N_FPN_LEVELS, load_fusion_head, load_vision_encoder
from torch_export import _fix_inplace_detach


def export_vision_encoder() -> tuple[torch.Tensor, ...]:
    """Export the vision-encoder half. Returns real fpn feature tensors (not
    dummy random ones) so export_fusion_head() can trace against realistic
    shapes/values in the same process, without reloading the checkpoint."""
    vision = load_vision_encoder(DEVICE, dtype=torch.float32)
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    with torch.no_grad():
        fpn_tensors = vision(pixel_values)
        exported = torch.export.export(vision, (pixel_values,), strict=False)
        exported = _fix_inplace_detach(exported)
    torch.export.save(exported, str(VISION_FP32_PT2))
    print(f"  {VISION_FP32_PT2} ({VISION_FP32_PT2.stat().st_size / 1e9:.2f} GB)")

    del vision, exported
    return fpn_tensors


def export_fusion_head(fpn_tensors: tuple[torch.Tensor, ...]) -> None:
    fusion = load_fusion_head(DEVICE, dtype=torch.float32)
    input_ids = torch.randint(
        0, 32000, (1, CONTEXT_LENGTH), dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        exported = torch.export.export(fusion, (*fpn_tensors, input_ids), strict=False)
        exported = _fix_inplace_detach(exported)
    torch.export.save(exported, str(FUSION_FP32_PT2))
    print(f"  {FUSION_FP32_PT2} ({FUSION_FP32_PT2.stat().st_size / 1e9:.2f} GB)")

    del fusion, exported
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fpn_tensors = export_vision_encoder()
    assert len(fpn_tensors) == 2 * N_FPN_LEVELS
    export_fusion_head(fpn_tensors)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
