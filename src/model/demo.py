"""Visual demo on a KITTI frame: classify every labelled object and plot the result.

Left: the camera image with projected 3D boxes coloured by correctness.
Right: bird's-eye view of the object point clouds with predicted labels.

Usage::

    python -m multimodal_transformer.model.demo --checkpoint runs/default/best.pth \\
        [--sample-id 000123] [--save demo.png]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi, crop_image
from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, list_sample_ids
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_camera_box
from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.data_pipeline.sample_builder import (
    MultimodalSampleBuilder,
    build_image_transform,
)
from multimodal_transformer.model.inference_engine import ObjectClassifier, TorchBackend
from multimodal_transformer.utils.config import load_yaml
from multimodal_transformer.utils.geometry import box_edges


def run_demo(checkpoint: str, sample_id: Optional[str], save: Optional[str]) -> None:
    data_cfg = load_yaml("dataset.yaml")
    split_dir = Path(data_cfg["kitti_root"]) / "training"
    sample_id = sample_id or random.choice(list_sample_ids(split_dir))
    frame = KittiFrame(split_dir, sample_id)
    image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)

    backend = TorchBackend(checkpoint)
    class_names = backend.model.class_names or data_cfg["class_names"]
    classifier = ObjectClassifier(backend, class_names)
    builder = MultimodalSampleBuilder(
        PointCloudNormalizer(1024, data_cfg.get("intensity_scale", 1.0), seed=0),
        build_image_transform(),
    )

    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[2.5, 1])
    ax_img.imshow(image_rgb)
    for label in frame.labels:
        target = data_cfg["class_map"].get(label.type)
        if target is None:
            continue
        box = label.oriented_box()
        points = crop_points_in_camera_box(frame.points, box, frame.calibration)
        if points.shape[0] < data_cfg.get("min_points", 5):
            continue
        roi = box_to_image_roi(box.corners(), frame.calibration, image_rgb.shape[:2])
        patch = crop_image(image_rgb, roi) if roi else None
        pred = classifier.predict(builder.build_batch([patch], [points]))[0]

        color = "lime" if pred.class_name == target else "red"
        pixels, _ = frame.calibration.rect_to_image(box.corners())
        if np.all(np.isfinite(pixels)):
            for i, j in box_edges():
                ax_img.plot(*pixels[[i, j]].T, color=color, linewidth=1)
            ax_img.text(*pixels[4], f"{pred.class_name} {pred.confidence:.2f}", color=color,
                        fontsize=8, backgroundcolor="black")  # fmt: skip
        ax_bev.scatter(points[:, 1], points[:, 0], s=1, color=color)
        ax_bev.annotate(pred.class_name, (points[:, 1].mean(), points[:, 0].max()), fontsize=7)

    ax_img.set_title(f"KITTI {sample_id}: green = correct, red = wrong")
    ax_img.axis("off")
    ax_bev.set_title("Bird's-eye view of object points (LiDAR frame)")
    ax_bev.set_xlabel("y, m (left +)")
    ax_bev.set_ylabel("x, m (forward)")
    ax_bev.invert_xaxis()
    ax_bev.set_aspect("equal")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visual KITTI classification demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--save", help="save the figure instead of showing it")
    args = parser.parse_args(argv)
    run_demo(args.checkpoint, args.sample_id, args.save)


if __name__ == "__main__":
    main()
