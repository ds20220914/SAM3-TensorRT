"""Export the QDQ .pt2 graph to ONNX, via the dynamo exporter.

Run on the GPU box after quantize.py.
"""
import torch

from constant import CONTEXT_LENGTH, DEVICE, IMAGE_SIZE, QDQ_ONNX, QDQ_PT2


def export_to_onnx() -> None:
    model: torch.fx.GraphModule = torch.export.load(QDQ_PT2).module()
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    ids = torch.randint(0, 32000, (1, CONTEXT_LENGTH), dtype=torch.long, device=DEVICE)

    print(f"Exporting to {QDQ_ONNX.name}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (img, ids),
            str(QDQ_ONNX),
            input_names=["image", "tokenized_text"],
            output_names=["pred_masks", "pred_logits", "pred_boxes"],
            do_constant_folding=True,
            dynamo=True,
        )
    print(f"  {QDQ_ONNX} ({QDQ_ONNX.stat().st_size / 1e9:.2f} GB)")

    del model, img, ids
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    export_to_onnx()
