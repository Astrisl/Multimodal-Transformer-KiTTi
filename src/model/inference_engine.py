"""Backend-agnostic classifier used by the demo and the ROS 2 inference node.

Backends share one call signature, so switching from PyTorch (development PC) to
TensorRT (Jetson) is a config change (``inference.backend``), not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from multimodal_transformer.model.fusion_transformer import load_checkpoint


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray


class InferenceBackend(ABC):
    """Maps a batch of model inputs to logits [B, C] (float32 numpy)."""

    @abstractmethod
    def logits(self, batch: Dict[str, torch.Tensor]) -> np.ndarray: ...


class TorchBackend(InferenceBackend):
    def __init__(self, checkpoint: str, device: str = "cuda", half: bool = False) -> None:
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model = load_checkpoint(checkpoint, self.device)
        self.half = half and self.device == "cuda"
        if self.half:
            self.model = self.model.half()

    @torch.inference_mode()
    def logits(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        dtype = torch.float16 if self.half else torch.float32
        out = self.model(
            batch["image"].to(self.device, dtype),
            batch["points"].to(self.device, dtype),
            batch["image_valid"].to(self.device),
        )
        return out.float().cpu().numpy()


class OnnxRuntimeBackend(InferenceBackend):
    def __init__(self, onnx_path: str, providers: Sequence[str] = ()) -> None:
        import onnxruntime as ort  # noqa: PLC0415 - optional dependency

        self.session = ort.InferenceSession(
            onnx_path, providers=list(providers) or ort.get_available_providers()
        )

    def logits(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        feeds = {k: batch[k].numpy().astype(np.float32) for k in ("image", "points", "image_valid")}
        return self.session.run(["logits"], feeds)[0]


class TensorRTBackend(InferenceBackend):
    """Runs a serialized TensorRT engine (TensorRT >= 8.6 name-based I/O API).

    Device buffers are torch CUDA tensors, so no pycuda dependency is needed.
    """

    def __init__(self, engine_path: str, num_classes: int) -> None:
        import tensorrt as trt  # noqa: PLC0415 - optional, device-specific dependency

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires a CUDA device")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.num_classes = num_classes
        self.stream = torch.cuda.Stream()

    def logits(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        inputs = {
            name: batch[name].to("cuda", torch.float32).contiguous()
            for name in ("image", "points", "image_valid")
        }
        size = inputs["points"].shape[0]
        output = torch.empty(size, self.num_classes, device="cuda", dtype=torch.float32)
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.set_tensor_address("logits", output.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return output.cpu().numpy()


def create_backend(cfg: Dict, num_classes: int) -> InferenceBackend:
    """Factory for the ``inference`` config section."""
    kind = cfg.get("backend", "torch")
    if kind == "torch":
        return TorchBackend(cfg["checkpoint"], cfg.get("device", "cuda"), cfg.get("fp16", False))
    if kind == "onnxruntime":
        return OnnxRuntimeBackend(cfg["onnx"])
    if kind == "tensorrt":
        return TensorRTBackend(cfg["engine"], num_classes)
    raise ValueError(f"Unknown inference backend: {kind}")


class ObjectClassifier:
    """Turns backend logits into :class:`Prediction` objects."""

    def __init__(self, backend: InferenceBackend, class_names: Sequence[str]) -> None:
        self.backend = backend
        self.class_names = list(class_names)

    def predict(self, batch: Dict[str, torch.Tensor]) -> List[Prediction]:
        logits = self.backend.logits(batch)
        logits = logits - logits.max(axis=1, keepdims=True)  # stable softmax
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        ids = probs.argmax(axis=1)
        return [
            Prediction(int(i), self.class_names[int(i)], float(p[i]), p) for i, p in zip(ids, probs)
        ]
