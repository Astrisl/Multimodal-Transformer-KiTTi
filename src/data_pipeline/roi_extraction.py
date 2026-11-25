"""Live ROI extraction: raw scan -> object candidates with oriented boxes.

Pipeline (all NumPy/SciPy, no PCL dependency so it runs anywhere Python runs):

1. Range / height crop of the region of interest.
2. Voxel down-sampling (keeps clustering cost bounded on dense scans).
3. Ground removal with a RANSAC plane fit.
4. Euclidean clustering: points closer than ``cluster_tolerance`` are connected; the
   connected components of that graph are the clusters (same result as PCL's
   EuclideanClusterExtraction and DBSCAN with min_samples=1).
5. OBB fitting: yaw from PCA of the cluster footprint in the XY plane, extents from
   the cluster's min/max in the rotated frame. The box bottom is then extended down to
   the ground plane: ground removal cuts ~``ground_distance`` off every object, while
   KITTI training boxes reach the ground, and the classifier must see the same shapes.

Boxes are expressed in the LiDAR frame (x forward, y left, z up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_box
from multimodal_transformer.utils.geometry import OrientedBox


@dataclass
class RoiConfig:
    """Parameters of :class:`RoiExtractor` (mirrors ``roi_extraction`` in the ROS config)."""

    x_range: Tuple[float, float] = (0.0, 40.0)
    y_range: Tuple[float, float] = (-20.0, 20.0)
    z_range: Tuple[float, float] = (-3.0, 2.0)
    # Half-angle of the processed sector around +x (deg); 180 = all around. Limiting it to
    # the camera's field of view skips clusters that could only be classified by LiDAR.
    max_azimuth_deg: float = 180.0
    voxel_size: float = 0.1
    ground_distance: float = 0.2
    ransac_iterations: int = 60
    max_ground_tilt_deg: float = 15.0
    cluster_tolerance: float = 0.5
    min_cluster_points: int = 10
    max_cluster_points: int = 20000
    min_size: Tuple[float, float, float] = (0.2, 0.2, 0.4)
    max_size: Tuple[float, float, float] = (7.0, 3.5, 3.5)


@dataclass
class ObjectCandidate:
    """A cluster with its oriented box (LiDAR frame); ``points`` are its full-resolution
    points, cropped on first access (the live pipeline re-crops the scan itself, so the
    ROI node never pays for them)."""

    box: OrientedBox
    yaw: float
    size_lwh: np.ndarray
    region: np.ndarray = field(repr=False)

    @cached_property
    def points(self) -> np.ndarray:
        # Re-crop the *non-downsampled* region so the classifier gets full density
        # (training crops were not voxelised either). A small margin compensates for
        # the box being fitted on the voxelised cloud. Like a KITTI label box, the
        # crop also contains the few ground points under the object.
        margin_box = OrientedBox(self.box.center, self.box.rotation, self.box.half_extents + 0.05)
        return crop_points_in_box(self.region, margin_box)


class RoiExtractor:
    """Extracts object candidates from one LiDAR scan."""

    def __init__(self, config: Optional[RoiConfig] = None, seed: int = 0) -> None:
        self.cfg = config or RoiConfig()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ steps
    def crop_region(self, points: np.ndarray) -> np.ndarray:
        c = self.cfg
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= c.x_range[0]) & (x <= c.x_range[1])
            & (y >= c.y_range[0]) & (y <= c.y_range[1])
            & (z >= c.z_range[0]) & (z <= c.z_range[1])
        )  # fmt: skip
        if c.max_azimuth_deg < 180.0:
            mask &= np.abs(np.degrees(np.arctan2(y, x))) <= c.max_azimuth_deg
        return points[mask]

    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Keeps the first point of every occupied voxel."""
        if points.shape[0] == 0 or self.cfg.voxel_size <= 0:
            return points
        keys = np.floor(points[:, :3] / self.cfg.voxel_size).astype(np.int64)
        # One integer per voxel: np.unique on a 1-D key is ~10x faster than on rows.
        keys -= keys.min(axis=0)
        dims = keys.max(axis=0) + 1
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        _, first = np.unique(flat, return_index=True)
        return points[np.sort(first)]

    def ground_mask(self, points: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RANSAC plane fit.

        Returns:
            Mask of ground points and the plane ``(a, b, c, d)`` with ``ax+by+cz+d=0``
            and ``c > 0``, or None if no plane was found.

        Only near-horizontal planes (normal within ``max_ground_tilt_deg`` of +Z) are
        accepted, otherwise a large wall or the side of a bus could win the vote.
        """
        n = points.shape[0]
        best = np.zeros(n, dtype=bool)
        best_plane: Optional[np.ndarray] = None
        if n < 3:
            return best, best_plane
        xyz = points[:, :3]
        cos_limit = np.cos(np.deg2rad(self.cfg.max_ground_tilt_deg))
        # All hypotheses are drawn first (same random sequence as one-by-one) and scored
        # with one matrix product instead of a Python loop over the iterations.
        samples = np.array(
            [self._rng.choice(n, 3, replace=False) for _ in range(self.cfg.ransac_iterations)]
        )
        p0, p1, p2 = xyz[samples[:, 0]], xyz[samples[:, 1]], xyz[samples[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        norms = np.linalg.norm(normals, axis=1)
        keep = norms >= 1e-6
        normals, p0 = normals[keep] / norms[keep, None], p0[keep]
        keep = np.abs(normals[:, 2]) >= cos_limit
        normals, p0 = normals[keep], p0[keep]
        if len(normals) == 0:
            return best, best_plane
        offsets = np.einsum("ij,ij->i", p0, normals)
        counts = (np.abs(xyz @ normals.T - offsets) < self.cfg.ground_distance).sum(axis=0)
        k = int(np.argmax(counts))  # first of the best, as with a strict ">" in a loop
        if counts[k] == 0:
            return best, best_plane
        normal = normals[k]
        best = np.abs((xyz - p0[k]) @ normal) < self.cfg.ground_distance
        normal = normal if normal[2] > 0 else -normal
        best_plane = np.append(normal, -normal @ p0[k])
        return best, best_plane

    def cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """Euclidean clustering; returns index arrays, one per accepted cluster."""
        n = points.shape[0]
        if n == 0:
            return []
        pairs = cKDTree(points[:, :3]).query_pairs(
            self.cfg.cluster_tolerance, output_type="ndarray"
        )
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=n_comp)
        valid = np.where(
            (sizes >= self.cfg.min_cluster_points) & (sizes <= self.cfg.max_cluster_points)
        )[0]
        order = np.argsort(labels, kind="stable")
        starts = np.concatenate([[0], np.cumsum(sizes)])
        return [order[starts[c] : starts[c + 1]] for c in valid]

    @staticmethod
    def fit_obb(points: np.ndarray) -> Tuple[OrientedBox, float, np.ndarray]:
        """Fits a yaw-only oriented box to a cluster.

        The principal axis of the XY footprint gives the heading; length is defined as
        the larger horizontal extent, so yaw is ambiguous by pi (irrelevant for
        classification, the model is trained with random yaw rotations).
        """
        xyz = points[:, :3]
        xy = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        if xy.shape[0] >= 2 and np.ptp(xy, axis=0).max() > 1e-6:
            eigvals, eigvecs = np.linalg.eigh(np.cov(xy.T))
            major = eigvecs[:, np.argmax(eigvals)]
            yaw = float(np.arctan2(major[1], major[0]))
        else:
            yaw = 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        # Rotate into the box frame: local = R(yaw)^T @ p.
        local_x = xyz[:, 0] * c + xyz[:, 1] * s
        local_y = -xyz[:, 0] * s + xyz[:, 1] * c
        lo = np.array([local_x.min(), local_y.min(), xyz[:, 2].min()])
        hi = np.array([local_x.max(), local_y.max(), xyz[:, 2].max()])
        size = hi - lo
        mid = (lo + hi) / 2.0
        center = np.array([mid[0] * c - mid[1] * s, mid[0] * s + mid[1] * c, mid[2]])
        return OrientedBox.from_lidar(center, size, yaw), yaw, size

    @staticmethod
    def _extend_to_ground(
        box: OrientedBox, size: np.ndarray, plane: np.ndarray
    ) -> Tuple[OrientedBox, np.ndarray]:
        """Lowers the box bottom to the ground height below the box centre."""
        a, b, c, d = plane
        ground_z = -(a * box.center[0] + b * box.center[1] + d) / c
        top = box.center[2] + size[2] / 2.0
        bottom = box.center[2] - size[2] / 2.0
        if ground_z >= bottom:
            return box, size
        size = np.array([size[0], size[1], top - ground_z])
        center = np.array([box.center[0], box.center[1], (top + ground_z) / 2.0])
        return OrientedBox(center, box.rotation, size / 2.0), size

    # ------------------------------------------------------------- pipeline
    def extract(self, scan: np.ndarray) -> List[ObjectCandidate]:
        """Runs the full pipeline on a scan [N, >=3]."""
        region = self.crop_region(scan)
        sparse = self.voxel_downsample(region)
        ground, plane = self.ground_mask(sparse)
        objects = sparse[~ground]

        candidates: List[ObjectCandidate] = []
        min_size, max_size = np.asarray(self.cfg.min_size), np.asarray(self.cfg.max_size)
        for idx in self.cluster(objects):
            box, yaw, size = self.fit_obb(objects[idx])
            if plane is not None:
                box, size = self._extend_to_ground(box, size, plane)
            # Compare sorted horizontal extents so the check is independent of which
            # side PCA called "length".
            horiz = np.sort(size[:2])[::-1]
            dims = np.array([horiz[0], horiz[1], size[2]])
            if np.any(dims > max_size) or np.all(dims < min_size):
                continue
            candidates.append(ObjectCandidate(box=box, yaw=yaw, size_lwh=size, region=region))
        return candidates
