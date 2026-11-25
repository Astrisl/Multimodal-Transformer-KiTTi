"""Export of a trained checkpoint to ONNX and a TensorRT engine (FP16 / INT8-ready).

ONNX export runs anywhere; the TensorRT engine must be built *on the target device*
(engines are specific to the GPU and TensorRT version), e.g. on the Jetson::

    python -m multimodal_transformer.model.export --checkpoint best.pth --onnx model.onnx
    python -m multimodal_transformer.model.export --onnx model.onnx --engine model.engine --fp16

Equivalent ``trtexec`` command (ships with JetPack)::

    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \\
        --minShapes=image:1x3x224x224,points:1x1024x4,image_valid:1 \\
        --optShapes=image:8x3x224x224,points:8x1024x4,image_valid:8 \\
        --maxShapes=image:32x3x224x224,points:32x1024x4,image_valid:32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from multimodal_transformer.model.fusion_transformer import (
    MultimodalFusionTransformer,
    load_checkpoint,
)
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

INPUT_NAMES = ["image", "points", "image_valid"]
OUTPUT_NAMES = ["logits"]


def export_onnx(
    model: MultimodalFusionTransformer,
    onnx_path: str,
    num_points: int = 1024,
    image_size: int = 224,
    opset: int = 17,
    half: bool = False,
) -> Path:
    """Exports ``model`` with a dynamic batch dimension.

    ``image_valid`` is exported as float32 (0/1) rather than bool because float inputs
    are supported by every TensorRT / ONNX Runtime version we target.

    With ``half`` the graph, its inputs and output are FP16 (traced on the GPU). TensorRT
    10.12+ builds such a strongly typed network with FP16 kernels; it is the only way to
    get FP16 in TensorRT 11, which no longer has the FP16 builder flag.
    """
    device = "cuda" if half else "cpu"
    dtype = torch.float16 if half else torch.float32
    model = model.eval().to(device, dtype)
    dummy = (
        torch.randn(2, 3, image_size, image_size, device=device, dtype=dtype),
        torch.randn(2, num_points, model.config.point_features, device=device, dtype=dtype),
        torch.ones(2, device=device, dtype=dtype),
    )
    dynamic = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic,
        opset_version=opset,
        dynamo=False,
    )
    LOGGER.info("ONNX model written to %s", onnx_path)
    return Path(onnx_path)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_batch: int = 32,
    num_points: int = 1024,
    image_size: int = 224,
    workspace_gb: float = 1.0,
    opt_batch: int = 8,
) -> Path:
    """Builds a serialized TensorRT engine (TensorRT >= 8.6 Python API).

    Raises:
        ImportError: If TensorRT is not installed (it is part of JetPack on Jetson).
        RuntimeError: If parsing or building fails.
    """
    import tensorrt as trt  # noqa: PLC0415 - optional, device-specific dependency

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        LOGGER.warning("TensorRT %s has no FP16 flag: export the ONNX with --half", trt.__version__)

    profile = builder.create_optimization_profile()
    opt_batch = min(opt_batch, max_batch)
    shapes = {
        "image": lambda b: (b, 3, image_size, image_size),
        "points": lambda b: (b, num_points, 4),
        "image_valid": lambda b: (b,),
    }
    for name, shape in shapes.items():
        profile.set_shape(name, shape(1), shape(opt_batch), shape(max_batch))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    Path(engine_path).write_bytes(bytes(serialized))
    LOGGER.info("TensorRT engine written to %s (fp16=%s)", engine_path, fp16)
    return Path(engine_path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX / TensorRT")
    parser.add_argument("--checkpoint", help="trainer checkpoint (.pth)")
    parser.add_argument("--onnx", required=True, help="ONNX path (written or read)")
    parser.add_argument("--engine", help="TensorRT engine output path")
    parser.add_argument("--fp16", action="store_true", help="enable FP16 kernels")
    parser.add_argument("--half", action="store_true", help="export the ONNX graph in FP16")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--opt-batch", type=int, default=8, help="batch TensorRT tunes kernels for")
    args = parser.parse_args(argv)

    if args.checkpoint:
        export_onnx(load_checkpoint(args.checkpoint), args.onnx, half=args.half)
    if args.engine:
        build_tensorrt_engine(
            args.onnx,
            args.engine,
            fp16=args.fp16,
            max_batch=args.max_batch,
            opt_batch=args.opt_batch,
        )


if __name__ == "__main__":
    main()
