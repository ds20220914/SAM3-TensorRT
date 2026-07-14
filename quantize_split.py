"""INT8 PTQ of the two split SAM3 pieces (vision encoder / text+fusion head).
Split-pipeline equivalent of quantize.py — see JETSON_MIGRATION.md for why
(TensorRT builder memory on boards like Jetson Orin Nano).

Run after torch_export_split.py. Produces:
  artifacts/sam3_vision_<IMAGE_SIZE>_int8_qdq.pt2
  artifacts/sam3_fusion_<IMAGE_SIZE>_int8_qdq.pt2

The fusion half is calibrated on real fpn features, not random ones: each
calibration image is run through the fp32 (unquantized) vision encoder to
get realistic vision_embeds, which then calibrate the fusion head's
activation ranges. This keeps calibration data representative without
depending on the vision engine already being quantized/built first.

pip install "embedl-deploy[tensorrt]"  (https://docs.embedl.com)
"""
import torch
from torch import nn

from constant import (
    DEVICE,
    FUSION_FP32_PT2,
    FUSION_QDQ_PT2,
    IMAGE_SIZE,
    PROMPT,
    VISION_FP32_PT2,
    VISION_QDQ_PT2,
)
from model_utils import load_tokenizer, load_vision_encoder, tokenize_prompt
from quantize import _find_patch_embed_conv, load_calibration_images

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


def _quantize_graph(
    gm: torch.fx.GraphModule,
    example_args: tuple[torch.Tensor, ...],
    calib_fn,
    out_pt2_path,
    extra_skip: set[nn.Module] | None = None,
) -> None:
    """Shared transform + INT8-quantize + re-export routine for either half."""
    fused = transform(gm, example_args).model.eval().to(device=DEVICE, dtype=torch.float32)

    skip_modules: set[nn.Module] = set(extra_skip or ())
    quant_cfg = QuantConfig(
        activation=TensorQuantConfig(
            Precision.INT8, symmetric=True, per_channel=False,
            calibration_method=CalibrationMethod.MINMAX,
        ),
        weight=TensorQuantConfig(
            Precision.INT8, symmetric=True, per_channel=True,
            calibration_method=CalibrationMethod.MINMAX,
        ),
        skip=ModulesToSkip(stub=skip_modules, weight=skip_modules, smooth={nn.LayerNorm}),
    )

    qmodel = quantize(
        fused, args=example_args, config=quant_cfg, forward_loop=calib_fn, freeze_weights=True
    )

    print(f"Re-exporting QDQ -> {out_pt2_path.name}")
    qmodel.eval()
    with torch.no_grad():
        exported = torch.export.export(qmodel, example_args, strict=False)
    torch.export.save(exported, str(out_pt2_path))
    print(f"  {out_pt2_path} ({out_pt2_path.stat().st_size / 1e9:.2f} GB)")

    del fused, qmodel, exported
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def quantize_vision() -> list[torch.Tensor]:
    gm = torch.export.load(VISION_FP32_PT2).module()
    dummy_pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    stub_skip: set[nn.Module] = set()
    if (patch := _find_patch_embed_conv(gm)) is not None:
        stub_skip.add(patch)

    calib_imgs = load_calibration_images()

    def calib_fn(model_fn: nn.Module) -> None:
        model_fn.eval()
        for img in calib_imgs:
            with torch.no_grad():
                model_fn(img.unsqueeze(0).to(device=DEVICE, dtype=torch.float32))

    _quantize_graph(gm, (dummy_pixel_values,), calib_fn, VISION_QDQ_PT2, extra_skip=stub_skip)
    del gm, dummy_pixel_values
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return calib_imgs


def quantize_fusion(calib_imgs: list[torch.Tensor]) -> None:
    gm = torch.export.load(FUSION_FP32_PT2).module()
    tokenizer = load_tokenizer()
    input_ids = tokenize_prompt(tokenizer, PROMPT, DEVICE)

    # Real fpn features for calibration, from the fp32 (unquantized) vision
    # encoder — keeps calibration data realistic without needing the
    # already-quantized vision engine built first.
    vision_fp32 = load_vision_encoder(DEVICE, dtype=torch.float32)
    calib_fpn_pairs = []
    with torch.no_grad():
        for img in calib_imgs:
            calib_fpn_pairs.append(vision_fp32(img.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)))
    del vision_fp32
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    def calib_fn(model_fn: nn.Module) -> None:
        model_fn.eval()
        for fpn in calib_fpn_pairs:
            with torch.no_grad():
                model_fn(*fpn, input_ids)

    _quantize_graph(gm, (*calib_fpn_pairs[0], input_ids), calib_fn, FUSION_QDQ_PT2)
    del gm, calib_fpn_pairs, input_ids
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    calib_imgs = quantize_vision()
    quantize_fusion(calib_imgs)
