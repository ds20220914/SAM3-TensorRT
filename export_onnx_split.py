"""Export both QDQ .pt2 graphs (vision encoder, text+fusion head) to ONNX.

Split-pipeline equivalent of export_onnx.py — see JETSON_MIGRATION.md. Run
after quantize_split.py. Produces:
  artifacts/sam3_vision_<IMAGE_SIZE>_int8_qdq.onnx
  artifacts/sam3_fusion_<IMAGE_SIZE>_int8_qdq.onnx
"""
import torch

from constant import (
    CONTEXT_LENGTH,
    DEVICE,
    FUSION_QDQ_ONNX,
    FUSION_QDQ_PT2,
    IMAGE_SIZE,
    VISION_QDQ_ONNX,
    VISION_QDQ_PT2,
)
from model_utils import FPN_TENSOR_NAMES as _FPN_NAMES


def export_vision_to_onnx() -> tuple[torch.Tensor, ...]:
    """Returns real fpn tensors (run through the QDQ model) so
    export_fusion_to_onnx() can trace against realistic shapes in the same
    process, without re-loading the vision graph."""
    model: torch.fx.GraphModule = torch.export.load(VISION_QDQ_PT2).module()
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    print(f"Exporting to {VISION_QDQ_ONNX.name}")
    with torch.no_grad():
        fpn_tensors = model(img)
        torch.onnx.export(
            model,
            (img,),
            str(VISION_QDQ_ONNX),
            input_names=["image"],
            output_names=_FPN_NAMES,
            do_constant_folding=True,
            dynamo=True,
        )
    print(f"  {VISION_QDQ_ONNX} ({VISION_QDQ_ONNX.stat().st_size / 1e9:.2f} GB)")

    del model, img
    return fpn_tensors


def export_fusion_to_onnx(fpn_tensors: tuple[torch.Tensor, ...]) -> None:
    model: torch.fx.GraphModule = torch.export.load(FUSION_QDQ_PT2).module()
    ids = torch.randint(0, 32000, (1, CONTEXT_LENGTH), dtype=torch.long, device=DEVICE)

    print(f"Exporting to {FUSION_QDQ_ONNX.name}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (*fpn_tensors, ids),
            str(FUSION_QDQ_ONNX),
            input_names=[*_FPN_NAMES, "tokenized_text"],
            output_names=["pred_masks", "pred_logits", "pred_boxes"],
            do_constant_folding=True,
            dynamo=True,
        )
    print(f"  {FUSION_QDQ_ONNX} ({FUSION_QDQ_ONNX.stat().st_size / 1e9:.2f} GB)")

    del model, ids
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fpn_tensors = export_vision_to_onnx()
    export_fusion_to_onnx(fpn_tensors)
