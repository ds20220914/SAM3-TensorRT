"""Export the SAM3 detector (image PCS path) to a fp32 .pt2 file via torch.export.

Run this first, on the GPU box. Produces artifacts/sam3_resized_<IMAGE_SIZE>.pt2,
consumed by quantize.py.
"""
import torch

from constant import CONTEXT_LENGTH, DEVICE, FP32_PT2, IMAGE_SIZE
from model_utils import load_wrapped_detector


def _fix_inplace_detach(exported: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Swap in-place aten.detach_ nodes for the out-of-place aten.detach.

    SAM3's _prepare_multilevel_features does an in-place .detach_() on a view
    (of a lifted positional-embedding constant), which torch.fx.Interpreter-
    based tools can't replay ("Can't detach views in-place"). Only this one op
    is swapped — leaves every other op exactly as exported so downstream
    Conv/Linear pattern matching still sees what it expects.
    """
    gm = exported.graph_module
    n_fixed = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.detach_.default:
            node.target = torch.ops.aten.detach.default
            n_fixed += 1
    if n_fixed:
        gm.graph.lint()
        gm.recompile()
    print(f"  fixed {n_fixed} in-place detach_ node(s)")
    return exported


def export_sam3_fp32_pt2() -> None:
    wrapped = load_wrapped_detector(DEVICE, dtype=torch.float32)

    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    input_ids = torch.randint(
        0, 32000, (1, CONTEXT_LENGTH), dtype=torch.long, device=DEVICE
    )

    with torch.no_grad():
        exported = torch.export.export(wrapped, (pixel_values, input_ids), strict=False)
        exported = _fix_inplace_detach(exported)
    torch.export.save(exported, str(FP32_PT2))
    print(f"  {FP32_PT2} ({FP32_PT2.stat().st_size / 1e9:.2f} GB)")

    del wrapped, pixel_values, input_ids, exported
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    export_sam3_fp32_pt2()
