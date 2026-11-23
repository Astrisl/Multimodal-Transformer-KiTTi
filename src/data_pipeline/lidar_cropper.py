"""Extraction of an object's LiDAR points by its oriented 3D box (FR-02)."""

from __future__ import annotations

import numpy as np

from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.geometry import OrientedBox


def crop_points_in_box(points: np.ndarray, box: OrientedBox) -> np.ndarray:
    """Returns the rows of ``points`` ([N, >=3], same frame as ``box``) inside ``box``."""
    if points.shape[0] == 0:
        return points
    # Cheap axis-aligned pre-selection first: the exact oriented test is then run on a
    # few hundred points instead of the whole scan (~120k), with the same result.
    corners = box.corners()
    low, high = corners.min(axis=0), corners.max(axis=0)
    xyz = points[:, :3]
    near = np.flatnonzero(
        (xyz[:, 0] >= low[0])
        & (xyz[:, 0] <= high[0])
        & (xyz[:, 1] >= low[1])
        & (xyz[:, 1] <= high[1])
    )
    near = near[box.contains(xyz[near])]
    return points[near]


def crop_points_in_camera_box(
    points_lidar: np.ndarray, box_rect: OrientedBox, calibration: Calibration
) -> np.ndarray:
    """Crops LiDAR points by a box given in the rectified camera frame (KITTI labels).

    Points are transformed to the camera frame only for the inside test; the returned
    rows are the original LiDAR rows (x, y, z, intensity), so the model always sees
    LiDAR-frame coordinates both offline and live.
    """
    if points_lidar.shape[0] == 0:
        return points_lidar
    points_rect = calibration.lidar_to_rect(points_lidar[:, :3])
    return points_lidar[box_rect.contains(points_rect)]
