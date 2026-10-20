"""Oriented 3D bounding boxes (OBB): corners and point-in-box tests (FR-02).

Two box conventions occur in the project:

* KITTI labels live in the rectified *camera* frame (x right, y down, z forward).
  The box is rotated by ``ry`` around the vertical **Y** axis and ``location`` is the
  centre of the **bottom** face.
* Boxes produced by the live ROI extraction live in the *LiDAR* frame (x forward,
  y left, z up), rotated by ``yaw`` around the vertical **Z** axis, with ``center``
  at the geometric centre.

Both are reduced to the generic form ``(center, rotation, half_extents)`` so the
derotation algorithm below is written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix around the Y axis (KITTI camera frame convention)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix around the Z axis (LiDAR frame convention)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class OrientedBox:
    """Generic oriented box.

    Attributes:
        center: Geometric centre, shape [3].
        rotation: Box-to-world rotation [3, 3]; its columns are the box axes.
        half_extents: Half sizes along the box axes, shape [3].
    """

    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def from_kitti(
        cls, location: Tuple[float, float, float], dims_hwl: Tuple[float, float, float], ry: float
    ) -> "OrientedBox":
        """Builds a box from a KITTI label (camera frame).

        KITTI ``location`` is the bottom-face centre and the camera Y axis points
        down, so the geometric centre is ``h / 2`` *above*, i.e. at ``y - h / 2``.
        Local axes are (length, height, width) along (x, y, z).
        """
        h, w, length = dims_hwl
        center = np.asarray(location, dtype=np.float64) - np.array([0.0, h / 2.0, 0.0])
        return cls(center, rotation_y(ry), np.array([length, h, w]) / 2.0)

    @classmethod
    def from_lidar(
        cls, center: Tuple[float, float, float], dims_lwh: Tuple[float, float, float], yaw: float
    ) -> "OrientedBox":
        """Builds a box in the LiDAR frame. Local axes are (length, width, height)."""
        return cls(
            np.asarray(center, dtype=np.float64), rotation_z(yaw), np.asarray(dims_lwh) / 2.0
        )

    def to_local(self, points_xyz: np.ndarray) -> np.ndarray:
        """Expresses points [N, 3] in the box frame (centre at origin, axis aligned).

        Derotation: a world point is ``p = R @ p_local + c`` so ``p_local = R^T (p - c)``.
        For row vectors this is ``(p - c) @ R`` - no explicit transpose/inverse is
        needed and the result is exact for any yaw, including |yaw| > pi/2.
        """
        return (points_xyz[:, :3] - self.center) @ self.rotation

    def contains(self, points_xyz: np.ndarray) -> np.ndarray:
        """Boolean mask [N] of points inside the box (boundary inclusive)."""
        local = self.to_local(points_xyz)
        return np.all(np.abs(local) <= self.half_extents, axis=1)

    def corners(self) -> np.ndarray:
        """The 8 box corners [8, 3] in the world frame.

        Order: the 4 corners with local axis-1 = -half, then the 4 with +half, each set
        going around the box, which keeps edge drawing (see :func:`box_edges`) simple.
        """
        signs = np.array(
            [
                [1, -1, 1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, -1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        local = signs * self.half_extents
        return local @ self.rotation.T + self.center


def box_edges() -> Tuple[Tuple[int, int], ...]:
    """Corner index pairs forming the 12 edges for :meth:`OrientedBox.corners`."""
    return (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )  # fmt: skip
