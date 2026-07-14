# Migrating to NVIDIA Jetson Orin Nano

This pipeline was built and tested on an x86_64 box with an RTX 2080 Ti
(Turing, compute capability 7.5, TensorRT 10.0.1, inside the
`nvcr.io/nvidia/tensorrt:24.05-py3` container). Orin Nano is ARM64 (aarch64)
with an Ampere GPU (compute capability 8.7) and 8GB of memory **shared**
between CPU and GPU (not a separate VRAM pool like the 2080 Ti's 11GB).
Nothing here runs unmodified on Jetson — this is what has to change.

## What moves where

Split the pipeline: keep the heavy, one-time steps on the x86 box; only
rebuild the engine and run inference on the Jetson.

| Step | Where | Why |
|---|---|---|
| `torch_export.py` | x86 box (unchanged) | fp32 export used 6.77GB CPU RSS in our benchmark — tight on an 8GB *shared* Jetson memory pool |
| `quantize.py` | x86 box (unchanged) | same memory concern, plus needs `embedl-deploy` which has no stated ARM64 support |
| `export_onnx.py` | x86 box (unchanged) | produces the QDQ ONNX — this is the only artifact that needs to cross over |
| `build_engine.py` | **Jetson (rebuild required)** | TensorRT engines are hardware- and version-locked — the `.engine` file built on the 2080 Ti will not load on Orin's different SM architecture/TensorRT build |
| `infer.py` | **Jetson** | this is the actual deployment target |

Transfer only: the QDQ ONNX file (`artifacts/sam3_resized_*_int8_qdq.onnx`
plus its external-data sibling files in the same directory), `sam3/`'s
tokenizer files only (`tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `config.json` — *not* `model.safetensors`, since
the Jetson never loads the fp32 model), and the code files
(`constant.py`, `model_utils.py`, `build_engine.py`, `infer.py`).

## 1. Flash JetPack

Use JetPack 6.x (L4T R36.x, Ubuntu 22.04 base) — it's the current mainline
for the Orin family and ships CUDA 12.x + TensorRT 10.x, close to what this
project already uses on the 2080 Ti box. Flash via NVIDIA SDK Manager or the
pre-built SD-card/NVMe image, per the Orin Nano dev kit's official setup
guide.

Unlike a desktop Linux box, there's no separate "install an NVIDIA driver"
step — the GPU driver is baked into the L4T BSP that JetPack flashes. This
sidesteps the exact class of driver/CUDA-version mismatch that broke
`torch.cuda` on the current x86 box, *as long as you use JetPack-provided
CUDA/TensorRT rather than mixing in unrelated pip wheels*.

## 2. Verify the JetPack-provided stack

```bash
sudo apt show nvidia-jetpack
dpkg -l | grep -E "tensorrt|cuda-toolkit"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

TensorRT's Python bindings and the CUDA toolkit ship with JetPack and are
tied to the system Python (usually 3.10 on JetPack 6) — don't `pip install
tensorrt` separately, it'll conflict with the system one.

## 3. Install the remaining Python dependencies

```bash
# PyTorch: NOT the PyPI wheel — use NVIDIA's Jetson-specific build matching
# your JetPack/CUDA version (only needed here for tokenization via
# transformers, not for running the fp32 model — that stays on the x86 box):
# https://developer.download.nvidia.com/compute/redist/jp/  (pick the wheel
# matching your JetPack CUDA version, e.g. v60 for JetPack 6.0)
pip3 install --no-cache-dir <jetson-specific-torch-wheel-url>

pip3 install git+https://github.com/huggingface/transformers.git
pip3 install pycuda "numpy<2" pillow
```

`pycuda` compiles against the local CUDA toolkit at install time — if it
fails to find `nvcc`, add `export PATH=/usr/local/cuda/bin:$PATH` first.

## 4. Copy the artifacts over

```bash
# from the x86 box:
scp -r artifacts/sam3_resized_924_int8_qdq.onnx* jetson-host:/path/to/project/artifacts/
scp constant.py model_utils.py build_engine.py infer.py jetson-host:/path/to/project/
scp sam3/tokenizer*.json sam3/special_tokens_map.json sam3/config.json jetson-host:/path/to/project/sam3/
```

## 5. Rebuild the engine on the Jetson itself

```bash
python3 build_engine.py
```

Same INT8+FP16 flags as the x86 build — but consider lowering
`WORKSPACE_BYTES` in `build_engine.py` (currently 4 GiB) to something like
1–2 GiB, since that's carved out of the *shared* 8GB pool, not a dedicated
11GB VRAM budget like the 2080 Ti has.

## 6. Lock power/clocks before benchmarking

Jetson aggressively scales clocks down for power saving by default, which
makes latency numbers inconsistent run-to-run:

```bash
sudo nvpmodel -m 0      # max performance power mode (MAXN)
sudo jetson_clocks       # lock clocks at their max for the current mode
```

## What to expect

- **Correctness**: identical — it's the same ONNX graph, same INT8/FP16
  weights, TensorRT just recompiles the kernels for Orin's SM 8.7 instead of
  the 2080 Ti's SM 7.5.
- **Speed**: our benchmark measured 117.5 ms/image on the 2080 Ti (4352 CUDA
  cores, 616 GB/s memory bandwidth). Orin Nano (1024 CUDA cores, ~68–102 GB/s
  depending on the exact module) has roughly a fifth of the compute and a
  fraction of the bandwidth — expect something in the 400 ms–1 s range,
  varying a lot with power mode. Since this project is for still-image
  analysis rather than real-time video, that's likely still workable — worth
  re-measuring with `test.py`-style timing once you're on real hardware
  rather than assuming.
- **Memory**: our INT8+FP16 engine alone used ~2.1GB of GPU memory during
  inference. That leaves reasonable headroom in Orin Nano's 8GB shared pool
  as long as nothing else heavy runs concurrently — but the 4GB Orin Nano
  variant is probably too tight; use the 8GB module.
- **`IMAGE_SIZE=924`**: consider re-testing at a smaller size (448/616) on
  Jetson if 924 turns out to be too slow or memory-tight in practice — the
  whole pipeline from `torch_export.py` onward would need to be re-run for a
  different `IMAGE_SIZE`, same as any other resolution change on the x86 box.
