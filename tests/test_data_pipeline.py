"""FR-02/FR-03: cropping, normalisation, dataset building and loading."""

import csv

import numpy as np
import pytest
import torch

from multimodal_transformer.data_pipeline.dataset import KittiMultimodalDataset, split_by_scene
from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, ObjectLabel
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_camera_box
from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.data_pipeline.sample_builder import (
    MultimodalSampleBuilder,
    build_image_transform,
    to_rgb,
)

CLASSES = ["Car", "Pedestrian", "Cyclist"]


# ------------------------------------------------------------------ normalisation
@pytest.mark.parametrize("count", [1, 7, 1023, 1024, 5000])
def test_normalizer_output_shape_is_fixed(count):
    points = np.random.default_rng(0).normal(size=(count, 4)).astype(np.float32)
    out = PointCloudNormalizer(1024, seed=0)(points)
    assert out.shape == (1024, 4) and out.dtype == np.float32


def test_normalizer_centres_and_keeps_all_points_when_padding():
    points = np.array([[10.0, 5.0, 1.0, 0.5], [12.0, 5.0, 1.0, 0.7]])
    out = PointCloudNormalizer(16, seed=0)(points)
    assert {tuple(r) for r in out[:, :3].round(6)} == {(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)}
    np.testing.assert_allclose(out[:2, :3], [[-1, 0, 0], [1, 0, 0]])  # originals first


def test_normalizer_downsamples_without_duplicates():
    points = np.arange(4000 * 4, dtype=np.float32).reshape(4000, 4)
    out = PointCloudNormalizer(1024, seed=0)(points)
    assert len(np.unique(out[:, 3])) == 1024


def test_normalizer_handles_empty_and_xyz_only():
    assert not PointCloudNormalizer(8)(np.zeros((0, 4))).any()
    out = PointCloudNormalizer(8, intensity_scale=2.0, seed=0)(np.ones((3, 3)))
    assert (out[:, 3] == 0).all()


def test_normalizer_is_deterministic_with_seed():
    pts = np.random.default_rng(3).normal(size=(3000, 4))
    np.testing.assert_array_equal(
        PointCloudNormalizer(1024, seed=5)(pts), PointCloudNormalizer(1024, seed=5)(pts)
    )


# ------------------------------------------------------------------ KITTI reading
def test_label_parsing():
    label = ObjectLabel.from_line("Car 0.12 1 -1.5 100 120 200 220 1.5 1.6 3.9 -2.0 1.7 20.0 -1.57")
    assert label.type == "Car" and label.occlusion == 1
    assert label.dims_hwl == (1.5, 1.6, 3.9) and label.ry == pytest.approx(-1.57)
    with pytest.raises(ValueError):
        ObjectLabel.from_line("Car 0 0")


def test_obb_crop_recovers_object_points(synthetic_kitti):
    frame = KittiFrame(synthetic_kitti, "000000")
    for label in frame.labels:
        if label.type == "DontCare":
            continue
        crop = crop_points_in_camera_box(frame.points, label.oriented_box(), frame.calibration)
        # Each synthetic object has exactly 200 points sampled inside its rotated box.
        assert crop.shape == (200, 4)


# ---------------------------------------------------------------- dataset builder
def test_builder_writes_index_and_crops(synthetic_crops):
    with open(synthetic_crops / "index.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 30  # 10 frames x 3 objects, DontCare skipped
    assert {r["class_name"] for r in rows} == set(CLASSES)
    for row in rows:
        points = np.load(synthetic_crops / "lidar" / f"{row['name']}.npy")
        assert points.shape == (200, 4)
        assert (synthetic_crops / "images" / f"{row['name']}.png").exists() == (
            row["has_image"] == "1"
        )


def test_split_by_scene_is_disjoint_and_complete():
    ids = [f"{i:06d}" for i in range(100)] * 3
    split = split_by_scene(ids, 0.2, seed=1)
    assert not set(split["train"]) & set(split["val"])
    assert len(split["train"]) + len(split["val"]) == 100 and len(split["val"]) == 20


def test_dataset_items(synthetic_crops):
    train = KittiMultimodalDataset(str(synthetic_crops), CLASSES, "train", augment=True)
    val = KittiMultimodalDataset(str(synthetic_crops), CLASSES, "val")
    assert len(train) + len(val) == 30
    item = train[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["points"].shape == (1024, 4)
    assert item["image_valid"].item() in (0.0, 1.0)
    assert item["label"].dtype == torch.long
    assert sum(train.class_counts()) == len(train)


# ------------------------------------------------------------------ sample builder
def test_dark_image_is_treated_as_missing():
    builder = MultimodalSampleBuilder(
        PointCloudNormalizer(64, seed=0), build_image_transform(), min_image_brightness=15
    )
    pts = np.random.default_rng(0).normal(size=(50, 4))
    dark = np.full((40, 40, 3), 5, dtype=np.uint8)
    bright = np.full((40, 40, 3), 120, dtype=np.uint8)
    assert builder.build(dark, pts)["image_valid"].item() == 0.0
    assert builder.build(bright, pts)["image_valid"].item() == 1.0
    assert builder.build(None, pts)["image"].abs().sum() == 0
    batch = builder.build_batch([bright, None], [pts, pts])
    assert batch["image"].shape == (2, 3, 224, 224) and batch["image_valid"].tolist() == [1, 0]


def test_ir_frames_become_three_channel():
    ir16 = (np.arange(100, dtype=np.uint16) * 600).reshape(10, 10)
    rgb = to_rgb(ir16, "mono16")
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8 and rgb.max() == 255
