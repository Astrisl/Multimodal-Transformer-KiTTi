"""Shared fixtures: a real KITTI calibration and a tiny synthetic KITTI split on disk."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.geometry import OrientedBox

# calib/000000.txt of the KITTI object training split.
KITTI_CALIB_TEXT = """\
P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.070493e+02 0.000000e+00 6.040814e+02 -3.797842e+02 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03
P3: 7.070493e+02 0.000000e+00 6.040814e+02 -3.341081e+02 0.000000e+00 7.070493e+02 1.805066e+02 2.330660e+00 0.000000e+00 0.000000e+00 1.000000e+00 3.201153e-03
R0_rect: 9.999128e-01 1.009263e-02 -8.511932e-03 -1.012729e-02 9.999406e-01 -4.037671e-03 8.470675e-03 4.123522e-03 9.999556e-01
Tr_velo_to_cam: 6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999753e-01 6.931141e-03 -1.143899e-03 -3.321029e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01
"""  # noqa: E501

IMAGE_SHAPE = (375, 1242)

# (type, h, w, l, x, y, z, ry) in the rectified camera frame.
SYNTHETIC_OBJECTS = [
    ("Car", 1.5, 1.7, 4.0, -3.0, 1.7, 15.0, 0.4),
    ("Pedestrian", 1.8, 0.6, 0.8, 2.0, 1.7, 9.0, -2.5),
    ("Cyclist", 1.7, 0.6, 1.8, 4.0, 1.7, 12.0, 1.9),
    ("DontCare", -1, -1, -1, -1000, -1000, -1000, -10),
]


@pytest.fixture(scope="session")
def kitti_calibration(tmp_path_factory) -> Calibration:
    path = tmp_path_factory.mktemp("calib") / "000000.txt"
    path.write_text(KITTI_CALIB_TEXT)
    return Calibration.from_kitti_file(path)


def label_line(obj) -> str:
    kind, h, w, length, x, y, z, ry = obj
    return f"{kind} 0.00 0 0.0 0 0 50 50 {h} {w} {length} {x} {y} {z} {ry}"


def sample_points_in_box(box: OrientedBox, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform points strictly inside ``box`` (world frame of the box)."""
    local = rng.uniform(-0.95, 0.95, size=(count, 3)) * box.half_extents
    return local @ box.rotation.T + box.center


def write_synthetic_frame(split_dir: Path, sample_id: str, calib: Calibration, seed: int) -> None:
    """Writes one KITTI-format frame whose LiDAR points match its labels exactly."""
    rng = np.random.default_rng(seed)
    for folder in ("image_2", "velodyne", "calib", "label_2"):
        (split_dir / folder).mkdir(parents=True, exist_ok=True)

    clouds = []
    for obj in SYNTHETIC_OBJECTS:
        if obj[0] == "DontCare":
            continue
        box = OrientedBox.from_kitti(obj[4:7], obj[1:4], obj[7])
        rect = sample_points_in_box(box, 200, rng)
        clouds.append(calib.rect_to_lidar(rect))
    # Background points far away from every object.
    clouds.append(np.column_stack([rng.uniform(30, 60, 500), rng.uniform(-20, 20, 500),
                                   rng.uniform(-1.7, 1, 500)]))  # fmt: skip
    xyz = np.vstack(clouds)
    scan = np.column_stack([xyz, rng.uniform(0, 1, len(xyz))]).astype(np.float32)
    scan.tofile(split_dir / "velodyne" / f"{sample_id}.bin")

    image = rng.integers(40, 255, size=(*IMAGE_SHAPE, 3), dtype=np.uint8)
    cv2.imwrite(str(split_dir / "image_2" / f"{sample_id}.png"), image)
    (split_dir / "calib" / f"{sample_id}.txt").write_text(KITTI_CALIB_TEXT)
    (split_dir / "label_2" / f"{sample_id}.txt").write_text(
        "\n".join(label_line(o) for o in SYNTHETIC_OBJECTS) + "\n"
    )


@pytest.fixture(scope="session")
def synthetic_kitti(tmp_path_factory, kitti_calibration) -> Path:
    """A ``training/`` split with 10 frames."""
    split_dir = tmp_path_factory.mktemp("kitti") / "training"
    for i in range(10):
        write_synthetic_frame(split_dir, f"{i:06d}", kitti_calibration, seed=i)
    return split_dir


@pytest.fixture(scope="session")
def synthetic_crops(tmp_path_factory, synthetic_kitti) -> Path:
    from multimodal_transformer.data_pipeline.build_dataset import DatasetBuilder

    out = tmp_path_factory.mktemp("crops")
    class_map = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist"}
    DatasetBuilder(synthetic_kitti, out, class_map).run()
    return out
