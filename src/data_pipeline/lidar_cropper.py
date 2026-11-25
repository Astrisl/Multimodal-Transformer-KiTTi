"""Extraction of an object's LiDAR points by its oriented 3D box (FR-02)."""

from __future__ import annotations

from typing import List, Sequence

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


def crop_points_in_boxes(points: np.ndarray, boxes: Sequence[OrientedBox]) -> List[np.ndarray]:
    """:func:`crop_points_in_box` for many boxes of one scan, with identical results.

    The scan is sorted by x once; each box then tests only the slice of points within its
    x extent (found by binary search) instead of scanning all ~120k points per box.
    """
    if points.shape[0] == 0 or not boxes:
        return [points[:0] for _ in boxes]
    order = np.argsort(points[:, 0], kind="stable")
    xs = points[order, 0]
    crops = []
    for box in boxes:
        corners = box.corners()
        low, high = corners.min(axis=0), corners.max(axis=0)
        start = np.searchsorted(xs, low[0], side="left")
        stop = np.searchsorted(xs, high[0], side="right")
        near = order[start:stop]
        y = points[near, 1]
        near = near[(y >= low[1]) & (y <= high[1])]
        near = near[box.contains(points[near, :3])]
        crops.append(points[np.sort(near)])  # original row order, as crop_points_in_box
    return crops


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
