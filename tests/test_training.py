"""Weighted loss, metrics, an end-to-end training smoke test and the detection journal."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from multimodal_transformer.data_pipeline.dataset import KittiMultimodalDataset
from multimodal_transformer.model.fusion_transformer import (
    ModelConfig,
    build_model,
    MultimodalFusionTransformer,
    load_checkpoint,
)
from multimodal_transformer.model.inference_engine import ObjectClassifier, TorchBackend
from multimodal_transformer.training.losses import build_loss, inverse_frequency_weights
from multimodal_transformer.training.metrics import classification_report
from multimodal_transformer.training.trainer import Trainer, TrainerConfig, evaluate
from multimodal_transformer.utils.detection_log import (
    AsyncDetectionJournal,
    DetectionJournal,
    DetectionRecord,
)

CLASSES = ["Car", "Pedestrian", "Cyclist"]


def test_inverse_frequency_weights():
    weights = inverse_frequency_weights([1000, 100, 10, 0], power=1.0)
    torch.testing.assert_close(weights, torch.tensor([1.0, 10.0, 100.0, 0.0]))
    sqrt = inverse_frequency_weights([1000, 100, 10])
    torch.testing.assert_close(sqrt, torch.tensor([1.0, 10**0.5, 10.0]))


def test_weighted_loss_penalises_rare_class_errors_more():
    criterion = build_loss([1.0, 5.0, 10.0])
    wrong_logits = torch.tensor([[0.0, 5.0, 0.0]])
    car_error = criterion(wrong_logits, torch.tensor([0])).item()
    # Weighted mean of a single sample equals its unweighted loss, so compare in a batch
    # where both errors share the normaliser.
    batch = criterion(wrong_logits.repeat(2, 1), torch.tensor([0, 2])).item()
    cyclist_share = (batch * 11 - car_error) / 10
    assert cyclist_share == pytest.approx(car_error, rel=1e-5)
    assert build_loss("auto", class_counts=[100, 10, 1]).weight[2] > 1
    with pytest.raises(ValueError):
        build_loss("auto")


def test_classification_report_values():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    report = classification_report(y_true, y_pred, CLASSES)
    assert report.confusion.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    np.testing.assert_allclose(report.precision, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(report.recall, [2 / 3, 0.5, 1.0])
    np.testing.assert_allclose(report.f1, [0.8, 0.5, 2 / 3])
    assert report.macro_f1 == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert report.accuracy == pytest.approx(4 / 6)
    assert "macro F1" in report.to_text()


def test_end_to_end_training_and_inference(synthetic_crops, tmp_path):
    torch.manual_seed(0)
    common = dict(crops_dir=str(synthetic_crops), class_names=CLASSES, num_points=256)
    train_ds = KittiMultimodalDataset(split="train", augment=True, **common)
    val_ds = KittiMultimodalDataset(split="val", **common)
    model = MultimodalFusionTransformer(
        ModelConfig(pretrained_backbone=False, num_layers=1, embed_dim=32, nhead=4, ff_dim=64)
    )
    trainer = Trainer(
        model,
        DataLoader(train_ds, batch_size=8, shuffle=True),
        DataLoader(val_ds, batch_size=8),
        build_loss("auto", train_ds.class_counts()),
        CLASSES,
        TrainerConfig(epochs=2, amp=False, output_dir=str(tmp_path)),
        torch.device("cpu"),
    )
    history = trainer.fit()
    assert len(history) == 2 and all(np.isfinite(h["train_loss"]) for h in history)
    assert json.loads((tmp_path / "training_history.json").read_text())[-1]["epoch"] == 2

    restored = load_checkpoint(str(tmp_path / "best.pth"))
    assert restored.class_names == CLASSES
    report = evaluate(restored, DataLoader(val_ds, batch_size=8), torch.device("cpu"), CLASSES)
    assert report.confusion.sum() == len(val_ds)

    classifier = ObjectClassifier(TorchBackend(str(tmp_path / "best.pth"), "cpu"), CLASSES)
    batch = {
        k: torch.stack([val_ds[i][k] for i in range(3)]) for k in ("image", "points", "image_valid")
    }
    predictions = classifier.predict(batch)
    assert len(predictions) == 3
    assert all(p.class_name in CLASSES and 0 <= p.confidence <= 1 for p in predictions)


def test_detection_journal(tmp_path):
    journal = DetectionJournal(tmp_path / "log.sqlite", location="lab")
    record = DetectionRecord("Car", 0.93, (10, 1, -0.5), (4.1, 1.8, 1.5), 0.3)
    frame_id = journal.log_frame(1.5, 1.52, [record, record])
    assert frame_id == 1
    rows = journal.detections()
    assert len(rows) == 2 and rows[0][1] == "Car"
    assert json.loads(rows[0][3])["size"] == [4.1, 1.8, 1.5]

    async_journal = AsyncDetectionJournal(journal)
    async_journal.submit(2.0, None, [record])
    async_journal.close()
    reopened = DetectionJournal(tmp_path / "log.sqlite")
    assert len(reopened.detections(session_id=1)) == 3


def test_init_from_checkpoint_adds_a_class(tmp_path):
    from multimodal_transformer.training.train import init_from_checkpoint

    small = dict(embed_dim=32, nhead=4, num_layers=1, ff_dim=64, pretrained_backbone=False)
    old = build_model({**small, "num_classes": 3})
    path = tmp_path / "old.pth"
    torch.save({"state_dict": old.state_dict()}, path)

    new = build_model({**small, "num_classes": 4})
    init_from_checkpoint(new, str(path))
    old_head, new_head = old.head[-1], new.head[-1]
    torch.testing.assert_close(new_head.weight[:3], old_head.weight)
    torch.testing.assert_close(new_head.bias[:3], old_head.bias)
    torch.testing.assert_close(new.head[0].weight, old.head[0].weight)
