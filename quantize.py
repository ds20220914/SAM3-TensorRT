"""INT8 post-training quantization (PTQ) of the SAM3 detector.

Run on the GPU box after torch_export.py. Calibrates from
calibration_images/*.jpg (this project is image-only), reusing
model_utils.py's preprocess_image/tokenize_prompt.

pip install "embedl-deploy[tensorrt]"  (https://docs.embedl.com)

The [tensorrt] extra is supposed to pull in the companion embedl-deploy-tensorrt
package that actually provides the backend — if transform()/quantize() raise
"RuntimeError: No backends found", that companion package didn't install; check
with `pip show embedl-deploy-tensorrt` and install it directly if missing:
`pip install embedl-deploy-tensorrt`.
"""
import torch
from torch import nn

from constant import CALIB_PATH, DEVICE, CONTEXT_LENGTH, IMAGE_SIZE, N_CALIB, PROMPT, QDQ_PT2, FP32_PT2
from model_utils import load_tokenizer, preprocess_image, tokenize_prompt

from embedl_deploy import transform
from embedl_deploy.backend import set_backend
from embedl_deploy.quantize import (
    CalibrationMethod,
    ModulesToSkip,
    Precision,
    QuantConfig,
    TensorQuantConfig,
    quantize,
)

set_backend("tensorrt")


def _find_patch_embed_conv(model: nn.Module) -> nn.Conv2d | None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3 and max(m.kernel_size) >= 7:
            return m
    return None


def load_calibration_images() -> list[torch.Tensor]:
    images = sorted(CALIB_PATH.glob("*.jpg"))[:N_CALIB]
    if not images:
        raise FileNotFoundError(f"No calibration images found under {CALIB_PATH}/")
    print(f"  loaded {len(images)} calibration images from {CALIB_PATH}/")
    return [preprocess_image(p, DEVICE).squeeze(0) for p in images]


def quantize_to_qdq(gm: torch.fx.GraphModule) -> None:
    """Fuse and INT8-quantize the fp32 graph."""
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    ids = torch.randint(0, 32000, (1, CONTEXT_LENGTH), dtype=torch.long, device=DEVICE)
    fused = transform(gm, (img, ids)).model.eval().to(device=DEVICE, dtype=torch.float32)

    stub_w_skip: set[nn.Module] = set()
    if (patch := _find_patch_embed_conv(fused)) is not None:
        stub_w_skip.add(patch)

    quant_cfg = QuantConfig(
        activation=TensorQuantConfig(
            Precision.INT8, symmetric=True, per_channel=False,
            calibration_method=CalibrationMethod.MINMAX,
        ),
        weight=TensorQuantConfig(
            Precision.INT8, symmetric=True, per_channel=True,
            calibration_method=CalibrationMethod.MINMAX,
        ),
        skip=ModulesToSkip(stub=stub_w_skip, weight=stub_w_skip, smooth={nn.LayerNorm}),
    )

    tokenizer = load_tokenizer()
    input_ids = tokenize_prompt(tokenizer, PROMPT, DEVICE)
    calib_imgs = load_calibration_images()

    def calib_fn(model_fn: nn.Module) -> None:
        model_fn.eval()
        for calib_img in calib_imgs:
            with torch.no_grad():
                model_fn(calib_img.unsqueeze(0).to(device=DEVICE, dtype=torch.float32), input_ids)

    dummy_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    qmodel = quantize(
        fused,
        args=(dummy_img, input_ids),
        config=quant_cfg,
        forward_loop=calib_fn,
        freeze_weights=True,
    )

    del calib_imgs
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Re-exporting QDQ -> {QDQ_PT2.name}")
    qmodel.eval()
    with torch.no_grad():
        exported = torch.export.export(qmodel, (dummy_img, input_ids), strict=False)
    torch.export.save(exported, str(QDQ_PT2))
    print(f"  {QDQ_PT2} ({QDQ_PT2.stat().st_size / 1e9:.2f} GB)")

    del fused, qmodel, exported, dummy_img, input_ids
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    graph_module: torch.fx.GraphModule = torch.export.load(FP32_PT2).module()
    quantize_to_qdq(graph_module)
    del graph_module
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
