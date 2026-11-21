"""Experiments on a trained checkpoint that need no retraining (thesis sections 4.5-4.7).

* **Sparse LiDAR** (4.5): every validation object is evaluated with its cloud reduced to
  at most K points (K = 1024 ... 8) before normalisation, with the camera and without it.
* **Error analysis** (4.7): accuracy / macro F1 / per-class recall broken down by
  distance, number of LiDAR points, KITTI occlusion and truncation, plus the list of
  misclassified objects.
* **Speed** (4.6): latency and throughput of PyTorch (CUDA and CPU) and ONNX Runtime.

The validation split is read once: each object yields one image and one cloud per
density level, so all variants see exactly the same samples.

Usage::

    python -m multimodal_transformer.training.experiments \\
        --checkpoint runs/default/best.pth --onnx runs/default/model.onnx \\
        --output-dir runs/default/experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from multimodal_transformer.data_pipeline.dataset import KittiMultimodalDataset
from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.model.fusion_transformer import (
    MultimodalFusionTransformer,
    load_checkpoint,
)
from multimodal_transformer.training.metrics import classification_report
from multimodal_transformer.training.train import build_datasets
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# 0 = the object's full cloud (no reduction).
DEFAULT_LEVELS = (0, 1024, 512, 256, 128, 64, 32, 16, 8)
MODES = ("fused", "lidar")

# (column, label, bin edges as [low, high) or exact values)
BREAKDOWNS: Dict[str, List[Tuple[str, Callable[[float], bool]]]] = {
    "distance_m": [
        ("0-10 m", lambda v: v < 10),
        ("10-20 m", lambda v: 10 <= v < 20),
        ("20-30 m", lambda v: 20 <= v < 30),
        ("30-40 m", lambda v: 30 <= v < 40),
        (">=40 m", lambda v: v >= 40),
    ],
    "num_points": [
        ("5-15", lambda v: v < 16),
        ("16-63", lambda v: 16 <= v < 64),
        ("64-255", lambda v: 64 <= v < 256),
        ("256-1023", lambda v: 256 <= v < 1024),
        (">=1024", lambda v: v >= 1024),
    ],
    "occlusion": [
        ("0 fully visible", lambda v: v == 0),
        ("1 partly occluded", lambda v: v == 1),
        ("2 largely occluded", lambda v: v == 2),
        ("3 unknown", lambda v: v == 3),
    ],
    "truncation": [
        ("0", lambda v: v == 0),
        ("0-0.3", lambda v: 0 < v <= 0.3),
        (">0.3", lambda v: v > 0.3),
    ],
}


def level_name(level: int) -> str:
    return "all" if level == 0 else str(level)


class SparseLidarDataset(Dataset):
    """Wraps the validation dataset; returns the clouds of one object at every density.

    Reduction happens on the raw cloud (random subset of ``min(K, n)`` points), then
    the usual normaliser brings it back to ``num_points`` rows by duplication, exactly
    as a sparse cluster would be handled at run time. Seeds depend on the object
    index only, so results are reproducible and independent of the worker count.
    """

    def __init__(self, base: KittiMultimodalDataset, levels: Sequence[int], seed: int = 0) -> None:
        self.base = base
        self.levels = list(levels)
        self.seed = seed
        self.num_points = base.builder.normalizer.num_points
        self.intensity_scale = base.builder.normalizer.intensity_scale

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.base.rows[idx]
        raw = np.load(self.base.crops_dir / "lidar" / f"{row['name']}.npy")
        sample = self.base[idx]
        clouds = []
        for level in self.levels:
            rng = np.random.default_rng((self.seed, idx, level))
            pts = raw
            if level and len(raw) > level:
                pts = raw[rng.choice(len(raw), level, replace=False)]
            normalizer = PointCloudNormalizer(
                self.num_points, self.intensity_scale, seed=int(rng.integers(2**31))
            )
            clouds.append(torch.from_numpy(normalizer(pts)))
        return {
            "image": sample["image"],
            "image_valid": sample["image_valid"],
            "points": torch.stack(clouds),  # [L, N, 4]
            "label": sample["label"],
            "index": torch.tensor(idx),
        }


@torch.inference_mode()
def collect_predictions(
    model: MultimodalFusionTransformer,
    dataset: SparseLidarDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """Returns ``labels``, ``index`` and ``pred_<mode>_<level>`` arrays over the dataset."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    out: Dict[str, List[np.ndarray]] = {"labels": [], "index": []}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        valid = batch["image_valid"].to(device)
        points = batch["points"].to(device, non_blocking=True)
        out["labels"].append(batch["label"].numpy())
        out["index"].append(batch["index"].numpy())
        for li, level in enumerate(dataset.levels):
            for mode in MODES:
                v = valid if mode == "fused" else torch.zeros_like(valid)
                logits = model(image, points[:, li], v)
                key = f"pred_{mode}_{level_name(level)}"
                out.setdefault(key, []).append(logits.argmax(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def sparse_lidar_table(
    preds: Dict[str, np.ndarray],
    levels: Sequence[int],
    num_points: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, float]]:
    """One row per density level: metrics for both modes and the share of reduced objects."""
    rows = []
    for level in levels:
        name = level_name(level)
        row: Dict[str, float] = {
            "points": name,
            "reduced_share": float(np.mean(num_points > level)) if level else 0.0,
        }
        for mode in MODES:
            rep = classification_report(preds["labels"], preds[f"pred_{mode}_{name}"], class_names)
            row[f"{mode}_macro_f1"] = rep.macro_f1
            row[f"{mode}_accuracy"] = rep.accuracy
            for ci, cname in enumerate(class_names):
                row[f"{mode}_f1_{cname}"] = float(rep.f1[ci])
        rows.append(row)
    return rows


def breakdown_table(
    labels: np.ndarray,
    pred_fused: np.ndarray,
    pred_lidar: np.ndarray,
    values: np.ndarray,
    bins: List[Tuple[str, Callable[[float], bool]]],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Metrics of the objects falling into each bin of one attribute."""
    rows = []
    for label, test in bins:
        mask = np.array([test(v) for v in values], dtype=bool)
        if not mask.any():
            continue
        fused = classification_report(labels[mask], pred_fused[mask], class_names)
        lidar = classification_report(labels[mask], pred_lidar[mask], class_names)
        row: Dict[str, object] = {
            "bin": label,
            "count": int(mask.sum()),
            "errors_fused": int((labels[mask] != pred_fused[mask]).sum()),
            "fused_accuracy": fused.accuracy,
            "fused_macro_f1": fused.macro_f1,
            "lidar_accuracy": lidar.accuracy,
            "lidar_macro_f1": lidar.macro_f1,
        }
        for ci, cname in enumerate(class_names):
            row[f"count_{cname}"] = int(fused.support[ci])
            row[f"recall_{cname}"] = float(fused.recall[ci]) if fused.support[ci] else None
        rows.append(row)
    return rows


def _time_it(fn: Callable[[], object], warmup: int, iters: int, sync: Callable[[], None]) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000


@torch.inference_mode()
def benchmark_backends(
    checkpoint: str,
    onnx_path: Optional[str],
    batch_sizes: Sequence[int] = (1, 8, 32),
    num_points: int = 1024,
) -> List[Dict[str, object]]:
    """Latency (ms per batch), throughput (objects/s) and memory of each backend."""
    results = []
    gen = torch.Generator().manual_seed(0)

    def inputs(bs: int):
        return (
            torch.randn(bs, 3, 224, 224, generator=gen),
            torch.randn(bs, num_points, 4, generator=gen),
            torch.ones(bs),
        )

    backends = []
    if torch.cuda.is_available():
        backends.append(("PyTorch CUDA (FP32)", "cuda", False))
        backends.append(("PyTorch CUDA (FP16 autocast)", "cuda", True))
    backends.append(("PyTorch CPU", "cpu", False))

    for name, dev, fp16 in backends:
        device = torch.device(dev)
        model = load_checkpoint(checkpoint, dev).eval()
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        for bs in batch_sizes:
            x = [t.to(device) for t in inputs(bs)]
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()

            def run(model=model, x=x, fp16=fp16):
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    return model(*x)

            iters = 50 if dev == "cuda" else max(3, 40 // bs)
            ms = _time_it(run, warmup=5 if dev == "cuda" else 2, iters=iters, sync=sync)
            mem = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
            results.append(_speed_row(name, bs, ms, mem))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if onnx_path and Path(onnx_path).exists():
        import onnxruntime as ort

        for provider in ort.get_available_providers():
            if provider not in ("CUDAExecutionProvider", "CPUExecutionProvider"):
                continue
            sess = ort.InferenceSession(onnx_path, providers=[provider])
            label = "ONNX Runtime " + ("CUDA" if provider.startswith("CUDA") else "CPU")
            for bs in batch_sizes:
                image, points, valid = (t.numpy() for t in inputs(bs))
                feed = {"image": image, "points": points, "image_valid": valid}
                iters = 50 if provider.startswith("CUDA") else max(3, 40 // bs)
                ms = _time_it(lambda: sess.run(None, feed), 2, iters, lambda: None)
                results.append(_speed_row(label, bs, ms, None))
    return results


def _speed_row(backend: str, bs: int, ms: float, mem_mb: Optional[float]) -> Dict[str, object]:
    return {
        "backend": backend,
        "batch": bs,
        "latency_ms": ms,
        "objects_per_s": bs / ms * 1000,
        "gpu_peak_mem_mb": mem_mb,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if v <= 1 else f"{v:.1f}"
    return str(v)


def markdown_table(rows: List[Dict[str, object]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def plot_sparse(rows: List[Dict[str, float]], class_names: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reduced = [r for r in rows if r["points"] != "all"]
    xs = [int(r["points"]) for r in reduced]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, style in (("fused", "-o"), ("lidar", "--s")):
        label = "RGB + LiDAR" if mode == "fused" else "LiDAR only"
        axes[0].plot(xs, [r[f"{mode}_macro_f1"] for r in reduced], style, label=label)
    for cname in class_names:
        axes[1].plot(xs, [r[f"fused_f1_{cname}"] for r in reduced], "-o", label=cname)
    for ax, title in zip(axes, ("Macro F1", "Per-class F1 (RGB + LiDAR)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.invert_xaxis()
        ax.set_xlabel("Max LiDAR points per object")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_breakdown(rows: List[Dict[str, object]], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['bin']}\n(n={r['count']})" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [r["fused_macro_f1"] for r in rows], 0.4, label="RGB + LiDAR")
    ax.bar(x + 0.2, [r["lidar_macro_f1"] for r in rows], 0.4, label="LiDAR only")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_accuracy_experiments(
    model: MultimodalFusionTransformer,
    val_ds: KittiMultimodalDataset,
    device: torch.device,
    out_dir: Path,
    levels: Sequence[int] = DEFAULT_LEVELS,
    batch_size: int = 32,
    num_workers: int = 4,
    plots: bool = True,
) -> Dict[str, object]:
    """Sparse-LiDAR and error-analysis experiments; writes CSV / PNG files to ``out_dir``."""
    class_names = model.class_names or val_ds.class_names
    sparse_ds = SparseLidarDataset(val_ds, levels)
    preds = collect_predictions(model, sparse_ds, device, batch_size, num_workers)
    rows = [val_ds.rows[i] for i in preds["index"]]
    labels = preds["labels"]

    num_points = np.array([int(r["num_points"]) for r in rows])
    sparse = sparse_lidar_table(preds, levels, num_points, class_names)
    write_csv(out_dir / "sparse_lidar.csv", sparse)

    pf, pl = preds["pred_fused_all"], preds["pred_lidar_all"]
    breakdowns = {}
    for column, bins in BREAKDOWNS.items():
        values = np.array([float(r[column]) for r in rows])
        table = breakdown_table(labels, pf, pl, values, bins, class_names)
        breakdowns[column] = table
        write_csv(out_dir / f"errors_by_{column}.csv", table)

    errors = [
        {
            "name": r["name"],
            "kitti_type": r["kitti_type"],
            "true": class_names[t],
            "pred_fused": class_names[f],
            "pred_lidar": class_names[lo],
            "num_points": r["num_points"],
            "distance_m": round(float(r["distance_m"]), 1),
            "occlusion": r["occlusion"],
            "truncation": r["truncation"],
            "has_image": r["has_image"],
        }
        for r, t, f, lo in zip(rows, labels, pf, pl)
        if t != f
    ]
    write_csv(out_dir / "errors_fused.csv", errors)

    if plots:
        plot_sparse(sparse, class_names, out_dir / "sparse_lidar.png")
        plot_breakdown(
            breakdowns["distance_m"], "Macro F1 by distance", out_dir / "f1_by_distance.png"
        )
        plot_breakdown(
            breakdowns["num_points"], "Macro F1 by LiDAR points on the object",
            out_dir / "f1_by_points.png",
        )  # fmt: skip
        plot_breakdown(
            breakdowns["occlusion"], "Macro F1 by occlusion", out_dir / "f1_by_occlusion.png"
        )
    return {"sparse_lidar": sparse, "breakdowns": breakdowns, "error_count": len(errors)}


def write_report(results: Dict[str, object], class_names: Sequence[str], path: Path) -> None:
    parts = ["# Experiments without retraining\n"]
    if "sparse_lidar" in results:
        cols = ["points", "reduced_share", "fused_macro_f1", "lidar_macro_f1"]
        cols += [f"fused_f1_{c}" for c in class_names]
        parts += ["## 4.5 Sparse LiDAR\n", markdown_table(results["sparse_lidar"], cols), ""]
        parts.append("`reduced_share` = share of objects that originally had more points.\n")
    if "breakdowns" in results:
        parts.append("## 4.7 Error analysis (full clouds)\n")
        for column, table in results["breakdowns"].items():
            cols = ["bin", "count", "errors_fused", "fused_accuracy", "fused_macro_f1"]
            cols += ["lidar_macro_f1"] + [f"recall_{c}" for c in class_names]
            parts += [f"### By {column}\n", markdown_table(table, cols), ""]
        parts.append(f"Misclassified objects (RGB + LiDAR): {results['error_count']}, "
                     "listed in `errors_fused.csv`.\n")  # fmt: skip
    if "speed" in results:
        cols = ["backend", "batch", "latency_ms", "objects_per_s", "gpu_peak_mem_mb"]
        parts += ["## 4.6 Inference speed\n", markdown_table(results["speed"], cols), ""]
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Experiments on a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", help="ONNX model for the speed benchmark")
    parser.add_argument("--dataset", default="dataset.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS),
                        help="max points per object; 0 = full cloud")  # fmt: skip
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, str(device))
    data_cfg = load_yaml(args.dataset)
    class_names = model.class_names or data_cfg["class_names"]

    results: Dict[str, object] = {}
    if not args.skip_accuracy:
        levels = sorted(set(args.levels), key=lambda v: (v != 0, -v))
        _, val_ds = build_datasets(data_cfg, num_points=1024, augment=False)
        LOGGER.info("Accuracy experiments on %d objects, levels %s", len(val_ds), levels)
        results.update(
            run_accuracy_experiments(
                model, val_ds, device, out_dir, levels, args.batch_size, args.num_workers
            )
        )
    if not args.skip_speed:
        del model
        LOGGER.info("Speed benchmark")
        results["speed"] = benchmark_backends(args.checkpoint, args.onnx)
        write_csv(out_dir / "speed.csv", results["speed"])

    # Merge with earlier runs so --skip-accuracy / --skip-speed only refresh their part.
    summary_path = out_dir / "experiments.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, class_names, out_dir / "report.md")
    LOGGER.info("Results written to %s", out_dir)


if __name__ == "__main__":
    main()
