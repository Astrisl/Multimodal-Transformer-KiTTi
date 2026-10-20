"""LiDAR -> camera calibration and projection (single source of truth, FR-01).

Every place in the project that moves points between the LiDAR frame, the rectified
camera frame and the image plane goes through :class:`Calibration`, so the projection
math exists exactly once (offline KITTI pipeline, ROS 2 nodes and tests).

Conventions follow the KITTI devkit::

    x_rect  = R0_rect @ Tr_velo_to_cam @ x_velo        (3D, rectified camera frame)
    uvw     = P2 @ [x_rect, 1]                          (homogeneous image coordinates)
    (u, v)  = (uvw[0] / uvw[2], uvw[1] / uvw[2])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import yaml

PathLike = Union[str, Path]

# Points closer than this to the camera plane are numerically unstable to project.
MIN_PROJECTION_DEPTH_M = 0.1


@dataclass
class Calibration:
    """Intrinsic and extrinsic calibration of one camera/LiDAR pair.

    Attributes:
        P: Camera projection matrix of the rectified camera, shape [3, 4].
        R0: Rectification rotation, shape [3, 3]. Identity for non-stereo rigs.
        Tr_lidar_to_cam: Rigid LiDAR -> camera transform, shape [3, 4].
    """

    P: np.ndarray
    R0: np.ndarray = field(default_factory=lambda: np.eye(3))
    Tr_lidar_to_cam: np.ndarray = field(default_factory=lambda: np.eye(4)[:3])

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=np.float64).reshape(3, 4)
        self.R0 = np.asarray(self.R0, dtype=np.float64).reshape(3, 3)
        self.Tr_lidar_to_cam = np.asarray(self.Tr_lidar_to_cam, dtype=np.float64).reshape(3, 4)
        # Pre-compose R0 @ Tr once: it is applied to every point of every scan.
        self._lidar_to_rect = self.R0 @ self.Tr_lidar_to_cam
        self._rect_to_lidar = _invert_rigid(self._lidar_to_rect)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_kitti_file(cls, path: PathLike) -> "Calibration":
        """Reads a KITTI ``calib/XXXXXX.txt`` file (uses P2, R0_rect, Tr_velo_to_cam).

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If one of the required matrices is missing.
        """
        values: Dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                values[key.strip()] = np.array([float(x) for x in raw.split()])
        return cls(P=values["P2"], R0=values["R0_rect"], Tr_lidar_to_cam=values["Tr_velo_to_cam"])

    @classmethod
    def from_yaml(cls, path: PathLike) -> "Calibration":
        """Reads a live-sensor calibration (see ``config/calibration_example.yaml``)."""
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(
            P=data["P"],
            R0=data.get("R0", np.eye(3)),
            Tr_lidar_to_cam=data["Tr_lidar_to_cam"],
        )

    # --------------------------------------------------------------- transforms
    def lidar_to_rect(self, points_xyz: np.ndarray) -> np.ndarray:
        """Transforms LiDAR points [N, 3] into the rectified camera frame [N, 3]."""
        return points_xyz[:, :3] @ self._lidar_to_rect[:, :3].T + self._lidar_to_rect[:, 3]

    def rect_to_lidar(self, points_rect: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`lidar_to_rect`."""
        return points_rect[:, :3] @ self._rect_to_lidar[:, :3].T + self._rect_to_lidar[:, 3]

    def rect_to_image(self, points_rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects rectified-camera points onto the image plane.

        Returns:
            Tuple of pixel coordinates [N, 2] and depths [N]. Pixels of points with
            depth <= MIN_PROJECTION_DEPTH_M are NaN; callers must filter them.
        """
        homo = np.hstack([points_rect[:, :3], np.ones((points_rect.shape[0], 1))])
        uvw = homo @ self.P.T
        depth = uvw[:, 2].copy()
        pixels = np.full((points_rect.shape[0], 2), np.nan)
        valid = depth > MIN_PROJECTION_DEPTH_M
        pixels[valid] = uvw[valid, :2] / uvw[valid, 2:3]
        return pixels, depth

    def lidar_to_image(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects LiDAR points directly onto the image plane (see :meth:`rect_to_image`)."""
        return self.rect_to_image(self.lidar_to_rect(points_xyz))


def _invert_rigid(transform: np.ndarray) -> np.ndarray:
    """Inverts a [3, 4] transform [R | t].

    R0 @ Tr is not exactly orthonormal in KITTI (R0 is a rectification), so a general
    3x3 inverse is used instead of the transpose.
    """
    rot_inv = np.linalg.inv(transform[:, :3])
    return np.hstack([rot_inv, (-rot_inv @ transform[:, 3]).reshape(3, 1)])
