"""Run image inference through the two split TensorRT engines (vision
encoder, text+fusion head) and save an overlay with boxes for every
detection above CONFIDENCE.

Split-pipeline equivalent of infer.py — see JETSON_MIGRATION.md. Run after
build_engine_split.py. Chains two TrtRunner instances: the vision engine's
output tensors (named via model_utils.FPN_TENSOR_NAMES) are copied
device->host->device into the fusion engine's inputs — the only real cost of
splitting vs. one fused engine, and small in absolute terms next to the
model's own compute.

pip install tensorrt pycuda pillow numpy
"""
import numpy as np
import torch

from constant import CONFIDENCE, FUSION_ENGINE, IMAGE_PATH, NMS_IOU_THRESHOLD, PROMPT, VISION_ENGINE
from infer import (
    TrtRunner,
    cxcywh_to_xyxy_abs,
    draw_boxes_and_save,
    mask_to_bbox_abs,
    nms,
    preprocess_image_np,
    sigmoid,
)
from model_utils import FPN_TENSOR_NAMES, load_tokenizer, tokenize_prompt


def main() -> None:
    image, original_size = preprocess_image_np(IMAGE_PATH)
    tokenizer = load_tokenizer()
    tokenized_text = tokenize_prompt(tokenizer, PROMPT, torch.device("cpu")).numpy()

    print(f"Loading {VISION_ENGINE.name}...")
    vision_runner = TrtRunner(VISION_ENGINE)
    vision_outputs = vision_runner.infer({"image": image})
    fpn_feeds = {name: vision_outputs[name] for name in FPN_TENSOR_NAMES}

    print(f"Loading {FUSION_ENGINE.name}...")
    fusion_runner = TrtRunner(FUSION_ENGINE)
    outputs = fusion_runner.infer({**fpn_feeds, "tokenized_text": tokenized_text})

    # pred_logits is (batch, num_queries) — SAM3's PCS scores each query
    # against the single text concept directly, there's no separate class
    # dimension to max() over.
    logits = outputs["pred_logits"][0]  # (num_queries,)
    scores = sigmoid(logits)
    candidates = np.where(scores > CONFIDENCE)[0]
    print(f"  {len(candidates)} quer(y/ies) above confidence {CONFIDENCE}")

    boxes, kept_scores = [], []
    for idx in candidates:
        box = mask_to_bbox_abs(outputs["pred_masks"][0][idx], *original_size)
        if box is None:
            box = cxcywh_to_xyxy_abs(outputs["pred_boxes"][0][idx], *original_size)
        boxes.append(box)
        kept_scores.append(float(scores[idx]))

    keep = nms(boxes, kept_scores, NMS_IOU_THRESHOLD)
    boxes = [boxes[i] for i in keep]
    kept_scores = [kept_scores[i] for i in keep]
    print(f"  {len(boxes)} box(es) after NMS: {[round(s, 3) for s in kept_scores]}")

    out_path = IMAGE_PATH.parent / f"{IMAGE_PATH.stem}_result.jpg"
    draw_boxes_and_save(IMAGE_PATH, boxes, kept_scores, out_path)


if __name__ == "__main__":
    main()
