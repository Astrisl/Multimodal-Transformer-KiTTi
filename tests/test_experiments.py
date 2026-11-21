"""Sparse-LiDAR and error-analysis experiments on the synthetic KITTI split."""

import numpy as np
import torch

from multimodal_transformer.data_pipeline.dataset import KittiMultimodalDataset
from multimodal_transformer.model.fusion_transformer import ModelConfig, MultimodalFusionTransformer
from multimodal_transformer.training.experiments import (
    BREAKDOWNS,
    SparseLidarDataset,
    breakdown_table,
    run_accuracy_experiments,
    write_report,
)

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def _val_dataset(crops):
    return KittiMultimodalDataset(str(crops), CLASSES, split="all", num_points=64)


def test_sparse_dataset_limits_unique_points(synthetic_crops):
    ds = SparseLidarDataset(_val_dataset(synthetic_crops), levels=[0, 16, 4])
    item = ds[0]
    assert item["points"].shape == (3, 64, 4)
    unique = [len(np.unique(item["points"][i].numpy(), axis=0)) for i in range(3)]
    raw = int(ds.base.rows[0]["num_points"])
    assert unique[0] == min(raw, 64)
    assert unique[1] <= 16 and unique[2] <= 4
    # Same object, same level -> same cloud (reproducible across runs and workers).
    assert torch.equal(ds[0]["points"], item["points"])


def test_breakdown_table_bins_and_metrics():
    labels = np.array([0, 0, 1, 2])
    pred = np.array([0, 1, 1, 2])
    values = np.array([5.0, 15.0, 15.0, 50.0])
    rows = breakdown_table(labels, pred, labels, values, BREAKDOWNS["distance_m"], CLASSES)
    assert [r["bin"] for r in rows] == ["0-10 m", "10-20 m", ">=40 m"]
    assert [r["count"] for r in rows] == [1, 2, 1]
    assert rows[1]["errors_fused"] == 1 and rows[1]["fused_accuracy"] == 0.5
    assert rows[1]["lidar_accuracy"] == 1.0


def test_accuracy_experiments_write_results(synthetic_crops, tmp_path):
    torch.manual_seed(0)
    model = MultimodalFusionTransformer(
        ModelConfig(pretrained_backbone=False, num_layers=1, embed_dim=32, nhead=4, ff_dim=64)
    )
    model.class_names = CLASSES
    val_ds = _val_dataset(synthetic_crops)
    results = run_accuracy_experiments(
        model, val_ds, torch.device("cpu"), tmp_path, levels=[0, 32, 8],
        batch_size=4, num_workers=0, plots=True,
    )  # fmt: skip
    assert [r["points"] for r in results["sparse_lidar"]] == ["all", "32", "8"]
    total = sum(r["count"] for r in results["breakdowns"]["distance_m"])
    assert total == len(val_ds)
    write_report(results, CLASSES, tmp_path / "report.md")
    assert "Misclassified objects" in (tmp_path / "report.md").read_text(encoding="utf-8")
    for name in ("sparse_lidar.csv", "errors_by_distance_m.csv", "sparse_lidar.png", "report.md"):
        assert (tmp_path / name).exists(), name
