# SAM3 → INT8 TensorRT (image-only)

Quantizes SAM3's text-promptable detector (Promptable Concept Segmentation) to INT8
and builds a TensorRT engine for fast image inference. Video/tracking is unused —
only `Sam3VideoModel.detector_model` is exported.

Follows the tested recipe from
[embedl-deploy's SAM3 tutorial](https://docs.embedl.com/embedl-deploy/latest/auto_tutorials/sam3.html),
adapted for image-only calibration/inference (the tutorial calibrates from and
runs on video; this project calibrates from `calibration_images/*.jpg` and
runs on a single `images/bag.png`). We initially hand-rolled the PTQ step
with raw `nvidia-modelopt` before switching to `embedl-deploy` — see
"Why embedl-deploy" below if you're wondering why this looks different from a
typical modelopt pipeline.

This machine has no NVIDIA GPU, so nothing below runs here. Copy this folder
(including `sam3/`) to a Linux box with CUDA + TensorRT, `pip install -r
requirements.txt`, then run in order:

```
python torch_export.py   # fp32 detector -> artifacts/sam3_resized_924.pt2
python quantize.py       # INT8 PTQ (calibration_images/) -> ..._int8_qdq.pt2
python export_onnx.py    # QDQ .pt2 -> ..._int8_qdq.onnx (dynamo export)
python validate.py       # sanity-check ONNX vs eager fp32 on images/bag.png
python build_engine.py   # ONNX -> artifacts/sam3.engine (INT8+FP16 hybrid)
python infer.py          # run the engine on images/bag.png -> images/bag_result.jpg
```

## C++ engine builder

`cpp/build_engine.cpp` is a C++ port of `build_engine.py` (same INT8+FP16
flags, `BUILDER_OPTIMIZATION_LEVEL=3`, 4 GiB workspace, timing cache reuse,
plugin init) for deployment contexts where you don't want a Python dependency
at build time. It's a drop-in alternative to the `python build_engine.py` step
above — `quantize.py`/`export_onnx.py`/`validate.py`/`infer.py` are unchanged
either way.

```
cmake -B cpp/build -S cpp -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build cpp/build
./cpp/build/build_engine \
    artifacts/sam3_resized_924_int8_qdq.onnx \
    artifacts/sam3.engine \
    artifacts/trt_timing.cache \
    3
```

All four arguments are optional and default to the same paths `constant.py`
uses. Requires TensorRT >= 8.6 (for `setBuilderOptimizationLevel`).

## Before running

- **Model access**: `sam3/` must contain the real gated SAM3 weights (already
  present locally as `model.safetensors`).
- **transformers**: SAM3 requires a very recent dev build (`transformers_version:
  5.0.0.dev0` in `sam3/config.json`) — install from git, not PyPI stable.
- **Calibration images**: `calibration_images/*.jpg` should be representative of
  your real deployment images (same domain, e.g. actual cargo photos), not the
  synthetic placeholders currently in that folder — PTQ calibration quality
  directly determines INT8 accuracy. Swap in ~16-64 real images before running
  `quantize.py` for real.
- **Prompt**: `PROMPT` in `constant.py` (currently `"cargos"`) is baked into the
  calibration and inference tokenization — change it there if needed.
- **IMAGE_SIZE**: set to 924 in `constant.py` (the embedl-deploy tutorial's
  tested config). This uses more GPU/host memory during export than the 448
  used for the initial end-to-end smoke test — see gotchas below if export
  OOMs.

## Why embedl-deploy

We first tried quantizing directly with `nvidia-modelopt`, which led to a long
chain of export failures: `GuardOnDataDependentSymNode` from modelopt's
`TensorQuantizer` having plain-Python branches on calibrated values (not
`torch.export`-safe), which forced falling back to the legacy TorchScript ONNX
tracer, which then OOM'd on this model's size, which forced disabling constant
folding, which left un-folded dynamic `Slice` axes TensorRT's parser rejected,
and separately `torch.onnx.export` auto-quantizing Conv bias to INT32 in a way
this TensorRT version's `DequantizeLayer` rejects outright.

`embedl-deploy` is purpose-built and tested for exactly this SAM3 → TensorRT
path: its quantized modules are themselves `torch.export`-safe (no data-dependent
Python branches), so the *same* `torch.export.export(..., strict=False)` call
that already works for the plain fp32 model also works after quantization —
and the resulting ONNX export via the modern `dynamo=True` exporter doesn't
produce the dynamic-axes or INT32-bias problems the legacy tracer did. None of
our old `fold_onnx.py` graph-surgery workarounds are needed with this path.

## Other gotchas already worked around here

- **`_get_rpb_matrix` + SDPA vs. `torch.export`**: `model_utils.py` loads the
  model with `attn_implementation="eager"`. SAM3 extracts the vision feature
  spatial shape via `.item()` on a tensor instead of `.shape` inside
  `_get_rpb_matrix`, producing an unbacked symint; SDPA's internal
  shape-dispatch fast path then does a comparison against that symint that
  `torch.export`'s tracer can't resolve (`GuardOnDataDependentSymNode`). Eager
  attention skips that dispatch check entirely. This is a SAM3-modeling-code
  issue, unrelated to the modelopt-vs-embedl-deploy switch above, and still
  needed with embedl-deploy.
- **`sam3/onnx_weights/`** (~3.1 GB) is leftover data from a prior, incomplete
  ONNX export attempt (weight tensors with no matching graph file) — unrelated
  to this pipeline, safe to delete if you want the space back.
- **PyTorch refuses `torch.cuda` on this box** ("driver too old", even though
  the driver is new enough for TensorRT — proven by `build_engine.py` building
  successfully). This means `torch_export.py`/`quantize.py` silently run on
  CPU (`constant.DEVICE` falls back there when `torch.cuda.is_available()` is
  `False`) — correct, just slower, not worth fixing unless GPU-accelerated
  export/calibration matters to you (would need a matching NVIDIA driver
  upgrade or a torch wheel built for an older CUDA toolkit). `infer.py` uses
  `pycuda` instead of `torch.cuda` for its device buffers specifically to
  route around this — pycuda talks to the CUDA *driver* API directly, which
  doesn't have PyTorch's stricter version gate. Same root cause also breaks
  onnxruntime's `CUDAExecutionProvider` (`CUDNN_STATUS_NOT_INITIALIZED`), so
  `validate.py` runs CPU-only — it's a one-off sanity check, not a
  speed-critical path, so this doesn't matter.
