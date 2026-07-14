"""Sanity-check the exported/quantized artifacts against the eager fp32 model.

Compares the eager fp32 detector_model forward pass to the QDQ ONNX graph's
output (via onnxruntime, CPU or CUDA EP — no TensorRT engine needed for this
check) on a real image + prompt. Run this after quantize.py and export_onnx.py,
before build_engine.py: if the outputs diverge sharply here, the TensorRT
engine will too, and the bug is in export/quantization rather than the engine
build.

pip install onnxruntime-gpu  (or onnxruntime for CPU)
"""
import numpy as np
import onnxruntime as ort
import torch

from constant import DEVICE, IMAGE_PATH, PROMPT, QDQ_ONNX
from model_utils import load_tokenizer, load_wrapped_detector, preprocess_image, tokenize_prompt


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main() -> None:
    wrapped = load_wrapped_detector(DEVICE, dtype=torch.float32)
    tokenizer = load_tokenizer()

    pixel_values = preprocess_image(IMAGE_PATH, DEVICE)
    input_ids = tokenize_prompt(tokenizer, PROMPT, DEVICE)

    with torch.no_grad():
        eager_masks, eager_logits, eager_boxes = wrapped(pixel_values, input_ids)

    # CPU only — CUDAExecutionProvider fails to init cuDNN on this box (same
    # driver-too-old class of issue as torch.cuda; see README). This is a
    # one-off sanity check, not a speed-critical path, so CPU is fine.
    session = ort.InferenceSession(str(QDQ_ONNX), providers=["CPUExecutionProvider"])
    onnx_masks, onnx_logits, onnx_boxes = session.run(
        ["pred_masks", "pred_logits", "pred_boxes"],
        {
            "image": pixel_values.cpu().numpy(),
            "tokenized_text": input_ids.cpu().numpy(),
        },
    )

    print(f"  pred_masks  cosine sim: {cosine_sim(eager_masks.cpu().numpy(), onnx_masks):.4f}")
    print(f"  pred_logits cosine sim: {cosine_sim(eager_logits.cpu().numpy(), onnx_logits):.4f}")
    print(f"  pred_boxes  cosine sim: {cosine_sim(eager_boxes.cpu().numpy(), onnx_boxes):.4f}")

    # pred_logits is (batch, num_queries) — no separate class dim to max() over,
    # SAM3's PCS scores each query against the single text concept directly.
    eager_scores = torch.sigmoid(eager_logits)[0]
    top = torch.topk(eager_scores, k=min(5, eager_scores.shape[0]))
    print("  eager top scores:", [round(s, 3) for s in top.values.tolist()])
    print("  eager top boxes (cxcywh, normalized):")
    for idx in top.indices.tolist():
        print("   ", [round(v, 3) for v in eager_boxes[0, idx].tolist()])


if __name__ == "__main__":
    main()