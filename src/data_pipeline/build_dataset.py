"""Offline preprocessing: KITTI frames -> synchronised (image patch, LiDAR crop) pairs.

One pass over the dataset writes, for every labelled object::

    <output_dir>/images/<Class>_<sample>_<idx>.png   RGB patch around the projected box
    <output_dir>/lidar/<Class>_<sample>_<idx>.npy    float32 [N, 4] points inside the OBB
    <output_dir>/index.csv                          one row of metadata per object

Both modalities come from the *same* oriented box (v1 used an OBB for LiDAR in
training but an axis-aligned box at inference, which silently changed the input).

Usage::

    python -m multimodal_transformer.data_pipeline.build_dataset --config dataset.yaml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi, crop_image
from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, list_sample_ids
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_camera_box
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

INDEX_FIELDS = [
    "name", "sample_id", "object_idx", "kitti_type", "class_name", "num_points",
    "has_image", "occlusion", "truncation", "distance_m",
]  # fmt: skip


@dataclass
class IndexRow:
    name: str
    sample_id: str
    object_idx: int
    kitti_type: str
    class_name: str
    num_points: int
    has_image: int
    occlusion: int
    truncation: float
    distance_m: float


class DatasetBuilder:
    """Crops all labelled objects of a KITTI split.

    Args:
        split_dir: ``<kitti_root>/training``.
        output_dir: Destination directory.
        class_map: KITTI type -> project class; types not in the map are skipped.
        min_points: Objects with fewer LiDAR points are skipped (too far / occluded).
        image_padding: Context pixels around the projected box.
    """

    def __init__(
        self,
        split_dir: Path,
        output_dir: Path,
        class_map: Dict[str, str],
        min_points: int = 5,
        image_padding: int = 5,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.output_dir = Path(output_dir)
        self.class_map = class_map
        self.min_points = min_points
        self.image_padding = image_padding
        self.image_dir = self.output_dir / "images"
        self.lidar_dir = self.output_dir / "lidar"
        self.stats = {"objects": 0, "skipped_class": 0, "skipped_points": 0, "frame_errors": 0}

    def process_frame(self, sample_id: str) -> List[IndexRow]:
        frame = KittiFrame(self.split_dir, sample_id)
        image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
        rows: List[IndexRow] = []
        for idx, label in enumerate(frame.labels):
            class_name = self.class_map.get(label.type)
            if class_name is None:
                self.stats["skipped_class"] += 1
                continue
            box = label.oriented_box()
            points = crop_points_in_camera_box(frame.points, box, frame.calibration)
            if points.shape[0] < self.min_points:
                self.stats["skipped_points"] += 1
                continue

            name = f"{label.type}_{sample_id}_{idx}"
            np.save(self.lidar_dir / f"{name}.npy", points.astype(np.float32))
            roi = box_to_image_roi(
                box.corners(), frame.calibration, image_rgb.shape[:2], self.image_padding
            )
            if roi is not None:
                patch = crop_image(image_rgb, roi)
                cv2.imwrite(
                    str(self.image_dir / f"{name}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                )

            rows.append(
                IndexRow(
                    name=name,
                    sample_id=sample_id,
                    object_idx=idx,
                    kitti_type=label.type,
                    class_name=class_name,
                    num_points=int(points.shape[0]),
                    has_image=int(roi is not None),
                    occlusion=label.occlusion,
                    truncation=label.truncation,
                    distance_m=float(np.linalg.norm(label.location)),
                )
            )
            self.stats["objects"] += 1
        return rows

    def run(self, sample_ids: Optional[Iterable[str]] = None) -> Path:
        """Processes the split and writes ``index.csv``. Returns the index path."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.lidar_dir.mkdir(parents=True, exist_ok=True)
        ids = list(sample_ids) if sample_ids is not None else list_sample_ids(self.split_dir)

        index_path = self.output_dir / "index.csv"
        with open(index_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            for sample_id in tqdm(ids, desc="Cropping KITTI objects"):
                try:
                    rows = self.process_frame(sample_id)
                except (OSError, ValueError, KeyError) as exc:
                    # One corrupt frame must not abort a 7481-frame run.
                    self.stats["frame_errors"] += 1
                    LOGGER.error("Frame %s skipped: %s", sample_id, exc)
                    continue
                writer.writerows(asdict(row) for row in rows)
        LOGGER.info("Done: %s -> %s", self.stats, index_path)
        return index_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="dataset.yaml", help="dataset config (config/)")
    parser.add_argument("--limit", type=int, default=None, help="process only N frames")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    builder = DatasetBuilder(
        split_dir=Path(cfg["kitti_root"]) / "training",
        output_dir=Path(cfg["crops_dir"]),
        class_map=cfg["class_map"],
        min_points=cfg.get("min_points", 5),
        image_padding=cfg.get("image_padding", 5),
    )
    ids = list_sample_ids(builder.split_dir)
    builder.run(ids[: args.limit] if args.limit else ids)


if __name__ == "__main__":
    main()
