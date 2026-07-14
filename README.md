# SAM3 → INT8 TensorRT (image-only)

Quantizes SAM3's text-promptable detector (Promptable Concept Segmentation) to
INT8 and builds a TensorRT engine for fast image inference. Video/tracking is
unused — only `Sam3VideoModel.detector_model` is exported.

Tested on a Linux with CUDA + TensorRT and deployed on Nvidia Jetson Orin Nano Boards used by Mrdvs M4 mega camera systems, mainly for industrial depalletizing scenarios.

## Repo layout

```
constant.py          shared paths/config (IMAGE_SIZE, PROMPT, artifact paths, ...)
model_utils.py        model loading, preprocessing, tokenization
torch_export.py       fp32 export -> single .pt2
quantize.py            INT8 PTQ -> single QDQ .pt2
export_onnx.py         QDQ .pt2 -> single ONNX
build_engine.py         ONNX -> single TensorRT engine (Python)
validate.py             sanity-check fp32 vs ONNX outputs, no engine needed
infer.py                run the single engine on an image, draw boxes
test.py                 benchmark fp32 eager vs the single TensorRT engine
test_trt_only.py        benchmark a TensorRT engine alone (no torch needed)
cpp/build_engine.cpp    C++ port of build_engine.py

torch_export_split.py   fp32 export -> two .pt2 (vision encoder / fusion head)
quantize_split.py       INT8 PTQ -> two QDQ .pt2
export_onnx_split.py     two QDQ .pt2 -> two ONNX
build_engine_split.py    two ONNX -> two TensorRT engines
infer_split.py           chain both engines on an image, draw boxes

JETSON_MIGRATION.md     porting this to a Jetson Orin Nano (or similar)
```

There are two parallel pipelines: the **single-engine** path (one fused
model, simpler) and the **split** path (vision encoder and detection/mask
head as two separate engines, built and run independently). Use split only
if the single engine won't fit in a memory-constrained board's TensorRT
builder workspace — see JETSON_MIGRATION.md. Both produce numerically
equivalent results; `quantize_split.py` calibrates the fusion head on real
vision features (from the fp32 vision encoder), not random data.

## Prerequisites

- **Model access**: `sam3/` must contain the real gated SAM3 weights
  (`model.safetensors`, tokenizer files, `config.json`).
- **transformers**: SAM3 needs a very recent dev build (`transformers_version:
  5.0.0.dev0` in `sam3/config.json`) — install from git, not PyPI stable.
- **Calibration images**: `calibration_images/*.jpg` should match your real
  deployment domain — PTQ calibration quality directly determines INT8
  accuracy.
- **Prompt**: `PROMPT` in `constant.py` is baked into calibration and
  inference tokenization — change it there for your target object.
- **CUDA + TensorRT**: this whole pipeline needs an NVIDIA GPU with CUDA and
  TensorRT installed. Verify with `python3 -c "import tensorrt; print(tensorrt.__version__)"`.

## Deploying: single-engine path (Linux, step by step)

```bash
# 1. Get the code + model onto the target machine
git clone <this-repo-url> sam3_tensor_int8
cd sam3_tensor_int8
# copy the real gated weights into sam3/ (model.safetensors, tokenizer files,
# config.json) — not included in the git repo, too large / license-restricted

# 2. Install dependencies
python3 -m pip install -r requirements.txt

# 3. Set your target resolution and prompt
#    edit constant.py: IMAGE_SIZE, PROMPT, N_CALIB

# 4. Put representative calibration images in calibration_images/*.jpg
#    (16-100+ images matching your real deployment domain)

# 5. Put a test image at images/<name>.png and point IMAGE_PATH at it in
#    constant.py

# 6. Run the pipeline in order
python3 torch_export.py    # fp32 export -> artifacts/sam3_resized_<IMAGE_SIZE>.pt2
python3 quantize.py        # INT8 PTQ    -> artifacts/..._int8_qdq.pt2
python3 export_onnx.py     # ONNX export -> artifacts/..._int8_qdq.onnx
python3 validate.py        # sanity-check ONNX vs fp32 (optional but recommended)
python3 build_engine.py    # TensorRT engine -> artifacts/sam3.engine
python3 infer.py           # run inference -> images/<name>_result.jpg

# 7. (optional) benchmark
python3 test.py            # fp32 eager vs TensorRT engine, latency + memory
```

Each script prints the artifact path and size it produced — check those
before moving to the next step. `validate.py` prints cosine-similarity
numbers between fp32 and the quantized ONNX graph (should be ~0.95+) and the
top detection scores/boxes from the fp32 model, useful for catching
quantization or prompt-mismatch problems before spending time on
`build_engine.py`.

## Deploying: split-engine path (memory-constrained boards)

Same idea, two engines instead of one:

```bash
python3 torch_export_split.py   # -> artifacts/sam3_vision_*.pt2, sam3_fusion_*.pt2
python3 quantize_split.py       # -> artifacts/..._int8_qdq.pt2 (both halves)
python3 export_onnx_split.py    # -> artifacts/..._int8_qdq.onnx (both halves)
python3 build_engine_split.py   # -> artifacts/sam3_vision.engine, sam3_fusion.engine
python3 infer_split.py          # run inference -> images/<name>_result.jpg
```

`build_engine_split.py` (and `build_engine.py`) accept environment variable
overrides for TensorRT's builder workspace and optimization level, useful
when the default 4 GiB workspace doesn't fit:

```bash
TRT_WORKSPACE_MB=512 TRT_OPT_LEVEL=2 python3 build_engine_split.py
```

## C++ engine builder

`cpp/build_engine.cpp` is a C++ port of `build_engine.py` (same INT8+FP16
flags, `BUILDER_OPTIMIZATION_LEVEL`, workspace, timing cache reuse, plugin
init) — a drop-in alternative for deployment contexts where you don't want a
Python dependency at build time.

```bash
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

## Jetson / edge deployment

See [JETSON_MIGRATION.md](JETSON_MIGRATION.md) for porting to an ARM64 board
like a Jetson Orin Nano — TensorRT engines are hardware- and version-locked,
so the engine build step has to be redone on-device (or on matching
hardware); the export/quantize steps don't.

## Gotchas

- **`attn_implementation="eager"`**: `model_utils.py` loads the model with
  eager attention, not SDPA. SAM3 extracts the vision feature spatial shape
  via `.item()` on a tensor instead of `.shape` inside `_get_rpb_matrix`,
  producing an unbacked symint; SDPA's internal shape-dispatch fast path then
  does a comparison against that symint that `torch.export`'s tracer can't
  resolve (`GuardOnDataDependentSymNode`). Eager attention skips that
  dispatch check entirely.
- **ONNX external data**: `torch.onnx.export`'s dynamo exporter splits large
  models into a main `.onnx` graph file plus many small per-tensor weight
  files in the same directory (protobuf has a 2 GB single-file limit). This
  is normal — `build_engine.py`'s `parser.parse(bytes, path=...)` call
  resolves them automatically. Don't move the `.onnx` file without its
  sibling files.
- **PyTorch refusing `torch.cuda`**: on a box where the NVIDIA driver is
  older than what the installed PyTorch wheel expects, `torch.cuda.is_available()`
  silently returns `False` even though TensorRT itself works fine (proven by
  a successful `build_engine.py` run). `constant.DEVICE` falls back to CPU in
  that case — `torch_export.py`/`quantize.py` just run slower, not wrong.
  `infer.py`/`infer_split.py` use `pycuda` instead of `torch.cuda` for their
  device buffers specifically to route around this (pycuda talks to the CUDA
  driver API directly, without PyTorch's stricter version gate). Same root
  cause breaks onnxruntime's `CUDAExecutionProvider`, so `validate.py` runs
  CPU-only.
- **Box coordinates come from `pred_masks`, not `pred_boxes`**: SAM3's
  detector head doesn't regress a tight bounding box the way a DETR object
  detector does — `pred_boxes` is a loose/coarse localization box. `infer.py`
  derives a tight box by thresholding each query's `pred_masks` and taking
  its bounding rectangle instead.
