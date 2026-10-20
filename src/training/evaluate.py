"""Evaluation of a checkpoint against the requirements NFR-01..NFR-05.

Reports on the validation split:

* accuracy / per-class F1 / macro F1 with both modalities (NFR-03, target >= 0.88);
* the same with every image masked - LiDAR-only, i.e. darkness (NFR-05);
* inference latency per batch size and the derived throughput (NFR-01/02).

Usage::

    python -m multimodal_transformer.training.evaluate --checkpoint runs/default/best.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from multimodal_transformer.model.fusion_transformer import (
    MultimodalFusionTransformer,
    load_checkpoint,
)
from multimodal_transformer.training.train import build_datasets
from multimodal_transformer.training.trainer import evaluate
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@torch.inference_mode()
def benchmark_latency(
    model: MultimodalFusionTransformer,
    device: torch.device,
    batch_sizes=(1, 8),
    num_points: int = 1024,
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, float]:
    """Mean latency in ms of one forward pass, for each batch size."""
    model.eval()
    results = {}
    for bs in batch_sizes:
        image = torch.randn(bs, 3, 224, 224, device=device)
        points = torch.randn(bs, num_points, 4, device=device)
        valid = torch.ones(bs, device=device)
        for _ in range(warmup):
            model(image, points, valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(image, points, valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        results[f"latency_ms_batch{bs}"] = (time.perf_counter() - start) / iters * 1000
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="dataset.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args(argv)

    data_cfg = load_yaml(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, str(device))
    class_names = model.class_names or data_cfg["class_names"]

    _, val_ds = build_datasets(data_cfg, num_points=1024, augment=False)
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    fused = evaluate(model, loader, device, class_names)
    lidar_only = evaluate(model, loader, device, class_names, drop_image=True)
    latency = benchmark_latency(model, device)

    print("=== LiDAR + camera ===\n" + fused.to_text())
    print("=== LiDAR only (camera masked, NFR-05) ===\n" + lidar_only.to_text())
    print("=== Latency ===\n" + json.dumps(latency, indent=2))

    report = {
        "fused": fused.as_dict(),
        "lidar_only": lidar_only.as_dict(),
        "confusion_fused": fused.confusion.tolist(),
        "device": str(device),
        **latency,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
