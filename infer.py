"""Run image inference through the built TensorRT INT8+FP16 engine and save an overlay.

Run on the GPU box, after build_engine.py. Uses pycuda (not torch.cuda) for
device buffers: PyTorch on this box refuses to initialize CUDA ("driver too
old") even though the driver is new enough for TensorRT itself (build_engine.py
already proved this by successfully building an engine) — PyTorch's own
minimum-driver check is stricter than what its CUDA runtime actually needs.
pycuda talks to the CUDA *driver* API directly, bypassing that check
entirely, so it works here regardless. Tokenization still uses the HF
tokenizer, but on CPU only (model_utils.tokenize_prompt(..., device="cpu")) —
never touches torch.cuda.

Draws a red box around every detection scoring above CONFIDENCE (constant.py),
one per bag. Each box comes from thresholding that query's pred_masks (the
per-query segmentation mask) and taking its tight bounding rectangle, not from
pred_boxes directly: pred_boxes turned out to be a loose/coarse localization
box (SAM3's detector head doesn't regress a tight box the way a DETR object
detector would — pred_masks carries the real pixel-precise boundary).
SAM3's queries aren't deduplicated internally the way single-object DETR
decoders are, so multiple queries can fire on the same bag — NMS_IOU_THRESHOLD
(constant.py) drops the lower-scoring one whenever two boxes overlap past
that IoU.

pip install tensorrt pycuda pillow numpy
"""
import numpy as np
import pycuda.autoinit  # noqa: F401  (initializes a CUDA context via the driver API)
import pycuda.driver as cuda
import tensorrt as trt
import torch
from PIL import Image, ImageDraw

from constant import CONFIDENCE, ENGINE, IMAGE_PATH, IMAGE_SIZE, MEAN, NMS_IOU_THRESHOLD, PROMPT, STD
from model_utils import load_tokenizer, tokenize_prompt

_TRT_TO_NP = {
    trt.float32: np.float32,
    trt.float16: np.float16,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.int8: np.int8,
    trt.bool: np.bool_,
}


class TrtRunner:
    """TRT engine wrapper backed by pycuda device buffers."""

    def __init__(self, engine_path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.host_out: dict[str, np.ndarray] = {}
        self.device_bufs: dict[str, "cuda.DeviceAllocation"] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT_TO_NP[self.engine.get_tensor_dtype(name)]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.device_bufs[name] = cuda.mem_alloc(nbytes)
            self.ctx.set_tensor_address(name, int(self.device_bufs[name]))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.host_out[name] = np.empty(shape, dtype=dtype)
            else:
                self.input_names.append(name)

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self.input_names) - set(feeds)
        if missing:
            raise ValueError(f"infer() missing required input(s): {missing}")
        for name, arr in feeds.items():
            cuda.memcpy_htod_async(self.device_bufs[name], np.ascontiguousarray(arr), self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_out[name], self.device_bufs[name], self.stream)
        self.stream.synchronize()
        return self.host_out


def preprocess_image_np(path) -> tuple[np.ndarray, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size  # (W, H)
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0).numpy(), original_size


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def cxcywh_to_xyxy_abs(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
    x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
    return np.stack([x1, y1, x2, y2], axis=-1)


def mask_to_bbox_abs(mask: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Tight xyxy box (in original-image pixels) around a mask's foreground pixels.

    mask is raw logits (SAM convention: >0 means foreground, same threshold as
    sigmoid > 0.5) at the low-res mask resolution, not IMAGE_SIZE — scale up to
    the original image's own width/height.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    mask_h, mask_w = mask.shape
    scale_x, scale_y = width / mask_w, height / mask_h
    x1, x2 = xs.min() * scale_x, (xs.max() + 1) * scale_x
    y1, y2 = ys.min() * scale_y, (ys.max() + 1) * scale_y
    return np.array([x1, y1, x2, y2])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def nms(boxes: list[np.ndarray], scores: list[float], iou_threshold: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) < iou_threshold for j in keep):
            keep.append(i)
    return keep


def draw_boxes_and_save(image_path, boxes, scores, out_path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)
        draw.text((box[0], max(box[1] - 12, 0)), f"{score:.2f}", fill=(255, 0, 0))
    image.save(out_path)
    print(f"  saved {out_path} ({len(boxes)} box(es))")


def main() -> None:
    image, original_size = preprocess_image_np(IMAGE_PATH)
    tokenizer = load_tokenizer()
    tokenized_text = tokenize_prompt(tokenizer, PROMPT, torch.device("cpu")).numpy()

    runner = TrtRunner(ENGINE)
    outputs = runner.infer({"image": image, "tokenized_text": tokenized_text})

    # pred_logits is (batch, num_queries) — SAM3's PCS scores each query against
    # the single text concept directly, there's no separate class dimension to
    # max() over (that was the bug: it collapsed all 200 per-query scores down
    # to one scalar, which then produced garbage shapes downstream).
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
