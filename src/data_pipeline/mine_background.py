"""Hard-negative mining: adds a ``Background`` class built from the live candidate extractor.

The classifier was trained on crops of annotated objects only, so in the real-time
pipeline every cluster the ROI extractor returns (poles, kerbs, walls, bushes) is forced
into Car / Pedestrian / Cyclist, usually with a confidence close to 1. This script runs
the *same* :class:`RoiExtractor` as the ROS node on every KITTI Object training frame and
labels each candidate inside the camera view:

* mostly inside an annotated object of a project class -> that class (``roi_`` crops; they
  teach the model the boxes and partial clusters of the live extractor),
* not overlapping any annotation and not in a ``DontCare`` region -> ``Background``,
* anything ambiguous -> skipped.

Crops are cut exactly like the inference node does (LiDAR-frame box, image ROI of the
projected box) and written next to the existing crops. The output index
``index_with_background.csv`` = original rows + mined rows, so the frame-level
train / val split is unchanged.

    python -m multimodal_transformer.data_pipeline.mine_background --config dataset.yaml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from multimodal_transformer.data_pipeline.build_dataset import INDEX_FIELDS, IndexRow
from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi, crop_image
from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, ObjectLabel
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_box
from multimodal_transformer.data_pipeline.roi_extraction import RoiConfig, RoiExtractor
from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.geometry import OrientedBox

BACKGROUND = "Background"
INSIDE_OBJECT = 0.5  # share of candidate points inside an annotation -> that object
OUTSIDE_OBJECT = 0.1  # below this share (and outside DontCare) -> background
BOX_MARGIN = 0.3  # m added to annotation boxes before the overlap test

_STATE: Dict = {}


def label_box_lidar(label: ObjectLabel, calib: Calibration, margin: float = 0.0) -> OrientedBox:
    """Annotation box in the LiDAR frame (KITTI ry around camera y -> yaw around LiDAR z)."""
    box = label.oriented_box()
    center = calib.rect_to_lidar(box.center[None])[0]
    h, w, length = label.dims_hwl
    yaw = -label.ry - np.pi / 2.0
    return OrientedBox.from_lidar(center, (length + margin, w + margin, h + margin), yaw)


def in_camera(center_lidar: np.ndarray, calib: Calibration, shape) -> Optional[np.ndarray]:
    """Pixel of the centre if it is in front of the camera and inside the image."""
    rect = calib.lidar_to_rect(center_lidar[None])
    if rect[0, 2] <= 0.5:
        return None
    uv, _ = calib.rect_to_image(rect)
    u, v = uv[0]
    return uv[0] if (0 <= u < shape[1] and 0 <= v < shape[0]) else None


def _init_worker(cfg: Dict) -> None:
    _STATE.update(cfg)
    _STATE["extractor"] = RoiExtractor(RoiConfig(**cfg["roi"]))


def mine_frame(sample_id: str) -> List[Dict]:
    s = _STATE
    frame = KittiFrame(s["split_dir"], sample_id)
    calib = frame.calibration
    scan = frame.points
    image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
    shape = image_rgb.shape[:2]
    rng = np.random.default_rng([s["seed"], int(sample_id)])

    objects = [
        (label, label_box_lidar(label, calib, BOX_MARGIN))
        for label in frame.labels
        if label.type != "DontCare"
    ]
    dont_care = [label.bbox_2d for label in frame.labels if label.type == "DontCare"]

    background, positives = [], []
    for cand in s["extractor"].extract(scan):
        pixel = in_camera(cand.box.center, calib, shape)
        if pixel is None:  # KITTI annotates only the camera view
            continue
        points = crop_points_in_box(scan, cand.box)
        if points.shape[0] < s["min_points"]:
            continue
        overlap, best = 0.0, None
        for label, box in objects:
            share = float(box.contains(points[:, :3]).mean())
            if share > overlap:
                overlap, best = share, label
        if overlap >= INSIDE_OBJECT:
            cls = s["class_map"].get(best.type)
            if cls is not None:
                positives.append((cand, points, cls, best))
        elif overlap <= OUTSIDE_OBJECT:
            u, v = pixel
            if not any(x1 <= u <= x2 and y1 <= v <= y2 for x1, y1, x2, y2 in dont_care):
                background.append((cand, points, BACKGROUND, None))

    keep = []
    for group, cap in ((background, s["max_background"]), (positives, s["max_positives"])):
        order = rng.permutation(len(group))[:cap]
        keep += [group[i] for i in sorted(order)]

    rows = []
    for k, (cand, points, cls, label) in enumerate(keep):
        kitti_type = label.type if label is not None else BACKGROUND
        name = (
            f"{BACKGROUND}_{sample_id}_{k}"
            if label is None
            else f"roi_{kitti_type}_{sample_id}_{k}"
        )
        np.save(s["lidar_dir"] / f"{name}.npy", points.astype(np.float32))
        roi = box_to_image_roi(
            calib.lidar_to_rect(cand.box.corners()), calib, shape, s["image_padding"]
        )
        if roi is not None:
            patch = crop_image(image_rgb, roi)
            cv2.imwrite(str(s["image_dir"] / f"{name}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        rows.append(
            asdict(
                IndexRow(
                    name=name,
                    sample_id=sample_id,
                    object_idx=k,
                    kitti_type=kitti_type,
                    class_name=cls,
                    num_points=int(points.shape[0]),
                    has_image=int(roi is not None),
                    occlusion=label.occlusion if label is not None else 0,
                    truncation=label.truncation if label is not None else 0.0,
                    distance_m=float(np.linalg.norm(cand.box.center)),
                )
            )
        )
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="dataset.yaml")
    ap.add_argument("--ros-config", default="config/ros2_pipeline.yaml")
    ap.add_argument("--max-background", type=int, default=4, help="per frame")
    ap.add_argument("--max-positives", type=int, default=2, help="roi_ crops per frame")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    crops_dir = Path(cfg["crops_dir"])
    ros = yaml.safe_load(Path(args.ros_config).read_text(encoding="utf-8"))
    roi = ros["roi_extraction"]["ros__parameters"]
    roi = {
        k: tuple(v) if isinstance(v, list) else v for k, v in roi.items() if hasattr(RoiConfig, k)
    }
    split_dir = Path(cfg["kitti_root"]) / "training"
    state = {
        "split_dir": split_dir,
        "lidar_dir": crops_dir / "lidar",
        "image_dir": crops_dir / "images",
        "class_map": cfg["class_map"],
        "min_points": cfg.get("min_points", 5),
        "image_padding": cfg.get("image_padding", 5),
        "max_background": args.max_background,
        "max_positives": args.max_positives,
        "seed": args.seed,
        "roi": roi,
    }
    with open(crops_dir / "index.csv", "r", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    # Only frames that already have objects: the frame-level split is computed over the
    # frames of the index, so adding new frames would reshuffle train / val.
    ids = sorted({row["sample_id"] for row in original})[: args.limit]

    mined: List[Dict] = []
    with Pool(args.workers, initializer=_init_worker, initargs=(state,)) as pool:
        for rows in tqdm(pool.imap(mine_frame, ids, chunksize=8), total=len(ids), desc="Mining"):
            mined += rows

    out = crops_dir / "index_with_background.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(original)
        writer.writerows(mined)
    counts: Dict[str, int] = {}
    for row in mined:
        key = "roi_" + row["class_name"] if row["name"].startswith("roi_") else row["class_name"]
        counts[key] = counts.get(key, 0) + 1
    print(f"Mined {len(mined)} crops from {len(ids)} frames: {counts} -> {out}")


if __name__ == "__main__":
    main()
