"""Image-guided object candidates: a 2D detection selects the LiDAR points of one object.

Euclidean clustering (``roi_extraction.py``) merges a person with the wall, pole or car
they stand next to, so most pedestrians never reach the classifier. Here a 2D detector
finds people in the camera image and the object's points are taken from the *frustum* of
each detection (the idea of Frustum PointNets, Qi et al., CVPR 2018):

1. LiDAR points of the processed region are projected onto the image.
2. Among the non-ground points that fall into the central part of a 2D box, the densest
   interval of range (distance from the sensor) is the object: the person fills the
   middle of their own box, a wall behind them or a pole in front does not.
3. All points of the whole 2D box within that range interval form the candidate; an
   oriented box is fitted exactly as for a cluster.

The candidate is then classified by the multimodal model like any other, so a wrong 2D
detection can still be rejected as Background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from multimodal_transformer.data_pipeline.roi_extraction import (
    ObjectCandidate,
    RoiExtractor,
    Scene,
)
from multimodal_transformer.utils.calibration import Calibration

COCO_PERSON = 1


@dataclass
class ImageCandidateConfig:
    """Parameters of :func:`frustum_candidates`."""

    inner_fraction: float = 0.5  # central part of the 2D box (width) that votes for the range
    depth_window: float = 1.0  # m, range interval occupied by one person
    ground_clearance: float = 0.15  # m above the ground plane; lower points are ground
    max_height: float = 2.5  # m above the ground plane
    min_points: int = 5
    max_footprint: float = 2.5  # m, larger horizontal extents are not a person
    duplicate_distance: float = 0.6  # m between centres of the same object


class PersonDetector:
    """COCO-pretrained torchvision detector reduced to people.

    Returns boxes ``[K, 5]`` (x1, y1, x2, y2, score), best first. Weights are downloaded
    by torchvision on first use (set ``TORCH_HOME`` to choose where they are cached).
    """

    def __init__(
        self,
        device: str = "cuda",
        # Low on purpose: a wrong 2D box is still rejected by the multimodal classifier.
        score_threshold: float = 0.3,
        labels: Sequence[int] = (COCO_PERSON,),
    ) -> None:
        from torchvision.models import detection  # noqa: PLC0415 - heavy import

        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, box_score_thresh=score_threshold
        )
        self.model.eval().to(self.device)
        self.labels = torch.tensor(list(labels), device=self.device)

    @torch.inference_mode()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(np.ascontiguousarray(image_rgb)).to(self.device)
        out = self.model([image.permute(2, 0, 1).float().div(255)])[0]
        keep = torch.isin(out["labels"], self.labels)
        boxes = torch.cat([out["boxes"][keep], out["scores"][keep, None]], dim=1)
        return boxes.cpu().numpy()  # torchvision returns them sorted by score


def densest_range(ranges: np.ndarray, window: float) -> Tuple[float, int]:
    """Start of the ``window``-long interval holding most values, and their count.

    Of equally populated intervals the nearest wins.
    """
    order = np.sort(ranges)
    counts = np.searchsorted(order, order + window, side="right") - np.arange(order.size)
    best = int(np.argmax(counts))
    return float(order[best]), int(counts[best])


def frustum_candidates(
    scene: Scene,
    boxes_2d: np.ndarray,
    calibration: Calibration,
    config: Optional[ImageCandidateConfig] = None,
) -> List[ObjectCandidate]:
    """Candidates for 2D detections ``boxes_2d`` [K, >=4] (x1, y1, x2, y2), best first."""
    cfg = config or ImageCandidateConfig()
    region = scene.region
    if len(boxes_2d) == 0 or region.shape[0] == 0:
        return []
    xyz = region[:, :3]
    pixels, depth = calibration.lidar_to_image(xyz)
    usable = depth > 0
    if scene.plane is not None:
        height = xyz @ scene.plane[:3] + scene.plane[3]
        usable &= (height > cfg.ground_clearance) & (height < cfg.max_height)
    index = np.flatnonzero(usable)
    u, v = pixels[index, 0], pixels[index, 1]
    ranges = np.hypot(xyz[index, 0], xyz[index, 1])

    candidates: List[ObjectCandidate] = []
    for x1, y1, x2, y2 in np.asarray(boxes_2d)[:, :4]:
        in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        margin = (x2 - x1) * (1.0 - cfg.inner_fraction) / 2.0
        inner = in_box & (u >= x1 + margin) & (u <= x2 - margin)
        if inner.sum() < cfg.min_points:
            continue
        start, count = densest_range(ranges[inner], cfg.depth_window)
        if count < cfg.min_points:
            continue
        member = in_box & (ranges >= start) & (ranges <= start + cfg.depth_window)
        points = region[index[member]]
        box, yaw, size = RoiExtractor.fit_obb(points)
        if scene.plane is not None:
            box, size = RoiExtractor.extend_to_ground(box, size, scene.plane)
        if max(size[0], size[1]) > cfg.max_footprint:
            continue
        if any(
            np.linalg.norm(box.center[:2] - other.box.center[:2]) < cfg.duplicate_distance
            for other in candidates
        ):
            continue  # two detections of one person
        candidates.append(
            ObjectCandidate(box=box, yaw=yaw, size_lwh=size, region=region, source="image")
        )
    return candidates


def duplicate_losers(
    candidates: Sequence[ObjectCandidate],
    class_names: Sequence[str],
    rejected: Sequence[str] = ("Background",),
    config: Optional[ImageCandidateConfig] = None,
) -> List[int]:
    """Indices of classified candidates to drop because another one is the same object.

    An image-guided candidate and a cluster can cover one object. It is decided after
    classification: a cluster with an object class wins, because it holds the whole
    object (a cyclist's cluster includes the bicycle that the person's 2D box cuts off,
    so the image-guided candidate there looks like a pedestrian). A cluster called
    Background (a person merged with a wall) loses to the image-guided candidate.
    """
    cfg = config or ImageCandidateConfig()
    losers = set()
    for i, image_cand in enumerate(candidates):
        if image_cand.source != "image":
            continue
        for j, cluster in enumerate(candidates):
            if cluster.source == "image" or j in losers:
                continue
            reach = min(max(cluster.size_lwh[0], cluster.size_lwh[1]), cfg.max_footprint) / 2.0
            gap = np.linalg.norm(image_cand.box.center[:2] - cluster.box.center[:2])
            if gap > max(cfg.duplicate_distance, reach):
                continue
            if class_names[j] in rejected and class_names[i] not in rejected:
                losers.add(j)
            else:
                losers.add(i)
                break
    return sorted(losers)
