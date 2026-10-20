"""Normalisation of an object point cloud to a fixed tensor [num_points, 4] (FR-03)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class PointCloudNormalizer:
    """Centres a cloud and down-samples / pads it to exactly ``num_points`` rows.

    * Centring subtracts the centroid of x, y, z. Without it the network sees absolute
      positions and learns *where* objects usually are instead of their *shape*; the
      centroid is also identical for KITTI crops and live clusters, unlike a label box
      centre that does not exist at run time. Metric scale is kept on purpose: object
      size is the strongest LiDAR cue for Car vs Pedestrian.
    * Intensity is multiplied by ``intensity_scale`` so sensors with different ranges
      (KITTI reflectance 0..1, Livox reflectivity 0..255) share one input range.
    * More than ``num_points`` points -> random subset without replacement.
      Fewer -> all points kept + random duplicates, so no information is dropped and
      no artificial zero points are introduced.
    * Zero points -> an all-zero tensor (the caller decides whether to skip it).

    Args:
        num_points: Output row count.
        intensity_scale: Factor applied to column 3.
        seed: Fixed seed for deterministic sampling (evaluation / tests); None draws a
            fresh sample every call, which acts as augmentation during training.
    """

    def __init__(
        self, num_points: int = 1024, intensity_scale: float = 1.0, seed: Optional[int] = None
    ) -> None:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        self.num_points = num_points
        self.intensity_scale = intensity_scale
        self._rng = np.random.default_rng(seed)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Normalises ``points`` [N, 3 or 4] into float32 [num_points, 4]."""
        out = np.zeros((self.num_points, 4), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return out

        cloud = np.zeros((points.shape[0], 4), dtype=np.float32)
        cloud[:, :3] = points[:, :3]
        if points.shape[1] >= 4:
            cloud[:, 3] = points[:, 3] * self.intensity_scale
        cloud[:, :3] -= cloud[:, :3].mean(axis=0)

        count = cloud.shape[0]
        if count >= self.num_points:
            idx = self._rng.choice(count, self.num_points, replace=False)
        else:
            extra = self._rng.choice(count, self.num_points - count, replace=True)
            idx = np.concatenate([np.arange(count), extra])
        out[:] = cloud[idx]
        return out
