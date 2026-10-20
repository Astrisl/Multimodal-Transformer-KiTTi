"""Reading of the KITTI object detection dataset (``training/`` split).

Expected layout::

    <kitti_root>/training/
        image_2/000000.png     left colour camera
        velodyne/000000.bin    float32 [N, 4] = x, y, z, reflectance
        calib/000000.txt       P0..P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
        label_2/000000.txt     one object per line (15 columns)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.geometry import OrientedBox

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ObjectLabel:
    """One ground-truth object of a KITTI label file.

    Attributes:
        type: KITTI class name ('Car', 'Pedestrian', 'DontCare', ...).
        truncation: 0 (fully visible) .. 1 (leaving image boundaries).
        occlusion: 0 visible, 1 partly, 2 largely occluded, 3 unknown.
        bbox_2d: Left, top, right, bottom image coordinates.
        dims_hwl: Height, width, length in metres.
        location: Bottom-face centre in the rectified camera frame.
        ry: Rotation around the camera Y axis, [-pi, pi].
    """

    type: str
    truncation: float
    occlusion: int
    bbox_2d: tuple
    dims_hwl: tuple
    location: tuple
    ry: float

    @classmethod
    def from_line(cls, line: str) -> "ObjectLabel":
        cols = line.split()
        if len(cols) < 15:
            raise ValueError(f"KITTI label line must have >= 15 columns, got {len(cols)}")
        values = [float(v) for v in cols[1:15]]
        return cls(
            type=cols[0],
            truncation=values[0],
            occlusion=int(values[1]),
            bbox_2d=tuple(values[3:7]),
            dims_hwl=tuple(values[7:10]),
            location=tuple(values[10:13]),
            ry=values[13],
        )

    def oriented_box(self) -> OrientedBox:
        """Oriented 3D box in the rectified camera frame."""
        return OrientedBox.from_kitti(self.location, self.dims_hwl, self.ry)


def read_labels(path: PathLike) -> List[ObjectLabel]:
    """Parses a label file. A missing file (test split) yields an empty list."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [ObjectLabel.from_line(line) for line in handle if line.strip()]


def read_velodyne(path: PathLike) -> np.ndarray:
    """Reads a Velodyne scan as float32 [N, 4] (x, y, z, reflectance)."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


class KittiFrame:
    """Lazy accessor to all modalities of one KITTI frame."""

    def __init__(self, split_dir: PathLike, sample_id: str) -> None:
        self.split_dir = Path(split_dir)
        self.sample_id = sample_id
        self.calibration = Calibration.from_kitti_file(self._path("calib", ".txt"))
        self.labels = read_labels(self._path("label_2", ".txt"))
        self._image: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None

    def _path(self, folder: str, suffix: str) -> Path:
        return self.split_dir / folder / f"{self.sample_id}{suffix}"

    @property
    def image_bgr(self) -> np.ndarray:
        if self._image is None:
            image = cv2.imread(str(self._path("image_2", ".png")))
            if image is None:
                raise FileNotFoundError(self._path("image_2", ".png"))
            self._image = image
        return self._image

    @property
    def points(self) -> np.ndarray:
        if self._points is None:
            self._points = read_velodyne(self._path("velodyne", ".bin"))
        return self._points


def list_sample_ids(split_dir: PathLike) -> List[str]:
    """Sorted ids of frames that have a Velodyne scan.

    Raises:
        FileNotFoundError: If ``velodyne/`` is missing (wrong ``kitti_root``).
    """
    lidar_dir = Path(split_dir) / "velodyne"
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
    return sorted(p.stem for p in lidar_dir.glob("*.bin"))
