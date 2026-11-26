"""Offline replay of a KITTI raw drive through the real-time pipeline.

Runs, frame by frame, the same steps as the ROS 2 nodes (``roi_extraction`` ->
``multimodal_inference``) with the parameters of ``config/ros2_pipeline.yaml``, but without
ROS, so it works on any machine with the project installed. It measures

* the time of every stage (candidate extraction, sample building, model),
* how many annotated objects (KITTI tracklets) the clustering finds,
* how the found objects are classified, and how many confident labels are given to
  clutter (poles, walls, bushes) that is not an annotated object,

and renders a video: camera view with projected boxes on top, bird's-eye view below
(annotations as thin white boxes, predictions in class colours).

    python -m multimodal_transformer.training.drive_demo \\
        --drive data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from multimodal_transformer.data_pipeline.image_candidates import (
    PersonDetector,
    duplicate_losers,
    frustum_candidates,
)
from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi, crop_image
from multimodal_transformer.data_pipeline.kitti_raw import KittiRawDrive, Tracklet
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_boxes
from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.data_pipeline.roi_extraction import RoiConfig, RoiExtractor
from multimodal_transformer.data_pipeline.sample_builder import (
    MultimodalSampleBuilder,
    build_image_transform,
)
from multimodal_transformer.model.inference_engine import (
    ObjectClassifier,
    Prediction,
    create_backend,
)
from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.geometry import OrientedBox, box_edges
from multimodal_transformer.utils.tracking import ClassVotingTracker

CLASSES = ("Car", "Pedestrian", "Cyclist")
# Annotated types the classifier was not trained on; a prediction on them is neither
# right nor a false alarm (e.g. a Van called "Car").
IGNORED_TYPES = ("Van", "Truck", "Tram", "Misc", "Person_sitting")
COLORS_BGR = {"Car": (255, 153, 25), "Pedestrian": (51, 77, 255), "Cyclist": (25, 230, 255)}
REJECTED = ("Background",)  # predictions of these classes are not published
WARMUP_FRAMES = 3


@dataclass
class Detection:
    box: OrientedBox
    pred: Prediction
    in_camera: bool
    gt_type: Optional[str] = None  # annotated type of the matched tracklet
    source: str = "cluster"  # "image": candidate found through a 2D detection
    confirmed: bool = True  # False while its track is younger than the tracker's min_hits

    def published(self, min_conf: float) -> bool:
        return (
            self.confirmed
            and self.pred.confidence >= min_conf
            and self.pred.class_name not in REJECTED
        )


@dataclass
class GroundTruth:
    box: OrientedBox
    object_type: str
    num_points: int
    in_camera: bool
    found: bool = False


# ----------------------------------------------------------------------------- pipeline
class DrivePipeline:
    """The ROI extraction + inference node logic for one scan (no ROS)."""

    def __init__(
        self,
        params: Dict,
        calibration: Calibration,
        image_candidates: bool = False,
        person_score: float = 0.3,
    ) -> None:
        roi = {k: v for k, v in params["roi_extraction"].items() if hasattr(RoiConfig, k)}
        roi = {k: tuple(v) if isinstance(v, list) else v for k, v in roi.items()}
        self.extractor = RoiExtractor(RoiConfig(**roi))
        p = params["multimodal_inference"]
        self.p = p
        self.calibration = calibration
        self.builder = MultimodalSampleBuilder(
            normalizer=PointCloudNormalizer(p["num_points"], p["intensity_scale"]),
            image_transform=build_image_transform(),
            min_image_brightness=p["min_image_brightness"],
        )
        self.classifier = ObjectClassifier(
            create_backend(p, len(p["class_names"])), p["class_names"]
        )
        # People are also looked for in the image (clustering merges them with walls).
        self.detector = (
            PersonDetector(p.get("device", "cuda"), person_score) if image_candidates else None
        )

    def in_camera(self, box: OrientedBox, image_shape: Tuple[int, int]) -> bool:
        center = self.calibration.lidar_to_rect(box.center[None])
        if center[0, 2] <= 0.5:
            return False
        uv, _ = self.calibration.rect_to_image(center)
        u, v = uv[0]
        return 0 <= u < image_shape[1] and 0 <= v < image_shape[0]

    def run(self, scan: np.ndarray, image_rgb: Optional[np.ndarray]):
        """Returns (all classified candidates incl. low confidence, stage times in ms)."""
        t0 = time.perf_counter()
        scene = self.extractor.analyse(scan)
        candidates = scene.candidates
        t1 = time.perf_counter()
        if self.detector is not None and image_rgb is not None:
            people = self.detector(image_rgb)
            candidates = candidates + frustum_candidates(scene, people, self.calibration)
        image_s = time.perf_counter() - t1

        boxes, patches, clouds, kept = [], [], [], []
        crops = crop_points_in_boxes(scan, [cand.box for cand in candidates])
        for cand, points in zip(candidates, crops):
            if points.shape[0] < self.p["min_points"]:
                continue
            kept.append(cand)
            patch = None
            if image_rgb is not None:
                roi = box_to_image_roi(
                    self.calibration.lidar_to_rect(cand.box.corners()),
                    self.calibration,
                    image_rgb.shape[:2],
                    self.p["image_padding"],
                )
                patch = crop_image(image_rgb, roi) if roi else None
            boxes.append(cand.box)
            patches.append(patch)
            clouds.append(points)

        predictions: List[Prediction] = []
        prep_s = 0.0
        model_s = 0.0
        step = self.p["max_batch"]
        for i in range(0, len(clouds), step):
            s0 = time.perf_counter()
            batch = self.builder.build_batch(patches[i : i + step], clouds[i : i + step])
            s1 = time.perf_counter()
            predictions += self.classifier.predict(batch)
            prep_s += s1 - s0
            model_s += time.perf_counter() - s1
        t2 = time.perf_counter()

        shape = image_rgb.shape[:2] if image_rgb is not None else (375, 1242)
        detections = [
            Detection(box, pred, self.in_camera(box, shape), source=cand.source)
            for box, pred, cand in zip(boxes, predictions, kept)
        ]
        if self.detector is not None:
            losers = set(duplicate_losers(kept, [p.class_name for p in predictions], REJECTED))
            detections = [d for i, d in enumerate(detections) if i not in losers]
        times = {
            "roi_ms": 1000 * (t1 - t0),
            "image_ms": 1000 * image_s,
            "prep_ms": 1000 * (t2 - t1 - image_s - model_s),
            "model_ms": 1000 * model_s,
            "total_ms": 1000 * (t2 - t0),
        }
        return detections, times


def ground_truth(
    tracklets: List[Tracklet], frame: int, scan: np.ndarray, pipe: DrivePipeline, shape
) -> List[GroundTruth]:
    cfg = pipe.extractor.cfg
    out = []
    for track in tracklets:
        pose = track.pose_at(frame)
        if pose is None:
            continue
        (tx, ty, tz), yaw = pose
        h, w, length = track.size
        box = OrientedBox.from_lidar((tx, ty, tz + h / 2.0), (length, w, h), yaw)
        if not (cfg.x_range[0] <= tx <= cfg.x_range[1] and cfg.y_range[0] <= ty <= cfg.y_range[1]):
            continue
        n = int(box.contains(scan[:, :3]).sum())
        out.append(GroundTruth(box, track.object_type, n, pipe.in_camera(box, shape)))
    return out


def match(detections: List[Detection], gts: List[GroundTruth]) -> None:
    """Greedy nearest-centre matching in the ground plane (one detection per object)."""
    pairs = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(gts):
            dist = np.linalg.norm(det.box.center[:2] - gt.box.center[:2])
            # PCA boxes of half-seen cars are offset from the annotated centre.
            if dist <= max(1.0, gt.box.half_extents[0]):
                pairs.append((dist, di, gi))
    used_d, used_g = set(), set()
    for _, di, gi in sorted(pairs):
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        detections[di].gt_type = gts[gi].object_type
        gts[gi].found = True


# ------------------------------------------------------------------------------ drawing
def draw_camera(image_bgr, detections, calibration, min_conf):
    out = image_bgr.copy()
    for det in detections:
        if not det.published(min_conf) or not det.in_camera:
            continue
        color = COLORS_BGR[det.pred.class_name]
        uv, depth = calibration.lidar_to_image(det.box.corners())
        if np.any(depth <= 0.1):
            continue
        pts = uv.astype(int)
        for a, b in box_edges():
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        x, y = pts[:, 0].min(), max(pts[:, 1].min() - 6, 14)
        label = f"{det.pred.class_name} {det.pred.confidence:.2f}"
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_bev(scan, detections, gts, size, x_max, min_conf):
    w, h = size
    scale = h / x_max  # px per metre
    canvas = np.full((h, w, 3), 18, np.uint8)

    def to_px(xy):
        return np.stack([w / 2 - xy[:, 1] * scale, h - xy[:, 0] * scale], 1).astype(int)

    for r in range(10, int(x_max) + 1, 10):  # range rings
        cv2.circle(canvas, (w // 2, h), int(r * scale), (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{r} m",
            (w // 2 + 4, h - int(r * scale) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (90, 90, 90),
            1,
        )
    keep = (scan[:, 0] > 0) & (scan[:, 0] < x_max) & (np.abs(scan[:, 1]) < w / 2 / scale)
    px = to_px(scan[keep, :2])
    shade = np.clip(110 + 60 * scan[keep, 2], 60, 230).astype(np.uint8)
    canvas[np.clip(px[:, 1], 0, h - 1), np.clip(px[:, 0], 0, w - 1)] = shade[:, None]

    for gt in gts:
        poly = to_px(gt.box.corners()[[0, 3, 7, 4], :2])
        cv2.polylines(canvas, [poly], True, (220, 220, 220), 1, cv2.LINE_AA)
    for det in detections:
        if not det.published(min_conf):
            continue
        color = COLORS_BGR[det.pred.class_name]
        poly = to_px(det.box.corners()[[0, 3, 7, 4], :2])
        cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
        x, y = poly[:, 0].max() + 3, poly[:, 1].min() + 10
        cv2.putText(
            canvas,
            det.pred.class_name[:3],
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(canvas, (w // 2, h - 4), (0, 200, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return canvas


def compose(camera, bev, text_lines):
    frame = np.vstack([camera, bev])
    for i, line in enumerate(text_lines):
        y = camera.shape[0] + 22 + 20 * i
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
    return frame


# ----------------------------------------------------------------------------- run/report
def evaluate_drive(args) -> Dict:
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    params = {node: section["ros__parameters"] for node, section in raw.items()}
    inf = params["multimodal_inference"]
    if args.checkpoint:
        inf["checkpoint"] = args.checkpoint
    inf["backend"] = args.backend
    inf["device"] = args.device
    if args.onnx:
        inf["onnx"] = args.onnx
    if args.engine:
        inf["engine"] = args.engine
    if args.num_points:
        inf["num_points"] = args.num_points
    if args.intensity_scale is not None:
        inf["intensity_scale"] = args.intensity_scale
    min_conf = args.min_confidence if args.min_confidence is not None else inf["min_confidence"]

    drive = KittiRawDrive(args.drive)
    pipe = DrivePipeline(params, drive.calibration, args.image_candidates, args.person_score)
    tracker = ClassVotingTracker(min_hits=args.track_min_hits) if args.track else None
    names = pipe.classifier.class_names
    tracklets = drive.tracklets()
    out_dir = Path(args.output_dir) / drive.drive_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = []
    gt_stats = Counter()
    det_stats = Counter()
    confusion = Counter()
    frames = range(len(drive)) if args.max_frames is None else range(args.max_frames)
    for i in frames:
        scan = drive.points(i)
        image_bgr = drive.image_bgr(i)
        image_rgb = None if args.lidar_only else image_bgr[:, :, ::-1].copy()
        detections, times = pipe.run(scan, image_rgb)
        if tracker is not None and detections:
            pose = drive.lidar_poses[i] if drive.lidar_poses is not None else None
            tracked = tracker.update(
                np.array([d.box.center for d in detections]),
                np.array([d.pred.probabilities for d in detections]),
                pose,
            )
            for det, state in zip(detections, tracked):
                best = int(np.argmax(state.probabilities))
                det.pred = Prediction(
                    best, names[best], float(state.probabilities[best]), state.probabilities
                )
                det.confirmed = state.confirmed
        gts = ground_truth(tracklets, i, scan, pipe, image_bgr.shape[:2])
        match(detections, gts)

        for gt in gts:
            if gt.object_type in CLASSES and gt.in_camera and gt.num_points >= args.min_gt_points:
                gt_stats[(gt.object_type, "total")] += 1
                gt_stats[(gt.object_type, "found")] += gt.found
        for det in detections:
            if not det.in_camera:
                continue
            confident = det.published(min_conf)
            if det.gt_type in CLASSES:
                confusion[(det.gt_type, det.pred.class_name if confident else "dropped")] += 1
            elif det.gt_type in IGNORED_TYPES:
                det_stats["ignored_type"] += 1
            elif confident:
                det_stats[f"clutter_{det.pred.class_name}"] += 1
            else:
                det_stats["clutter_dropped"] += 1

        published = [d for d in detections if d.published(min_conf)]
        rows.append(
            {
                "frame": i,
                "candidates": len(detections),
                "from_image": sum(d.source == "image" for d in detections),
                "published": len(published),
                **{k: round(v, 2) for k, v in times.items()},
            }
        )

        if not args.no_video:
            recent = [r["total_ms"] for r in rows[max(WARMUP_FRAMES, len(rows) - 10) :]]
            ms = float(np.mean(recent)) if recent else times["total_ms"]
            cam = draw_camera(image_bgr, detections, drive.calibration, min_conf)
            bev = draw_bev(scan, detections, gts, (cam.shape[1], 520), 45.0, min_conf)
            frame = compose(
                cam,
                bev,
                [
                    f"{drive.drive_dir.name}  frame {i + 1}/{len(drive)}",
                    f"{len(detections)} candidates, {len(published)} labelled, {ms:.0f} ms/frame",
                    "white: annotation   blue: Car   red: Pedestrian   yellow: Cyclist",
                ],
            )
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_dir / "drive_demo.mp4"), fourcc, args.fps, frame.shape[1::-1]
                )
            writer.write(frame)
            if i in args.snapshots:
                cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frame)
    if writer is not None:
        writer.release()

    timed = rows[WARMUP_FRAMES:] or rows
    timing = {
        key: {
            "mean": round(float(np.mean([r[key] for r in timed])), 1),
            "p95": round(float(np.percentile([r[key] for r in timed], 95)), 1),
        }
        for key in ("roi_ms", "image_ms", "prep_ms", "model_ms", "total_ms")
    }
    timing["candidates_mean"] = round(float(np.mean([r["candidates"] for r in rows])), 1)
    timing["from_image_mean"] = round(float(np.mean([r["from_image"] for r in rows])), 2)
    timing["fps"] = round(1000.0 / timing["total_ms"]["mean"], 1)

    recall = {
        c: {
            "objects": gt_stats[(c, "total")],
            "found": gt_stats[(c, "found")],
            "recall": (
                round(gt_stats[(c, "found")] / gt_stats[(c, "total")], 3)
                if gt_stats[(c, "total")]
                else None
            ),
        }
        for c in CLASSES
    }
    matched = sum(v for (g, p), v in confusion.items() if p != "dropped")
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    clutter_labelled = sum(
        v for k, v in det_stats.items() if k.startswith("clutter_") and k != "clutter_dropped"
    )
    summary = {
        "drive": drive.drive_dir.name,
        "frames": len(rows),
        "camera": not args.lidar_only,
        "image_candidates": args.image_candidates,
        "tracking": args.track,
        "backend": f"{args.backend}/{args.device}",
        "min_confidence": min_conf,
        "timing": timing,
        "extraction_recall": recall,
        "classification": {
            "matched_labelled": matched,
            "accuracy": round(correct / matched, 4) if matched else None,
            "confusion": {f"{g}->{p}": v for (g, p), v in sorted(confusion.items())},
        },
        "clutter": {
            "labelled_per_frame": round(clutter_labelled / max(len(rows), 1), 2),
            **dict(sorted(det_stats.items())),
        },
    }
    tag = "lidar_only" if args.lidar_only else "fused"
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_dir / f"frames_{tag}.csv", "w", encoding="utf-8") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drive", required=True, help="<raw>/<date>/<date>_drive_XXXX_sync")
    ap.add_argument("--config", default="config/ros2_pipeline.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--backend", default="torch", choices=["torch", "onnxruntime", "tensorrt"])
    ap.add_argument("--onnx", default=None, help="ONNX model for --backend onnxruntime")
    ap.add_argument("--engine", default=None, help="TensorRT engine for --backend tensorrt")
    ap.add_argument("--num-points", type=int, default=None, help="point tokens per object")
    ap.add_argument(
        "--intensity-scale",
        type=float,
        default=None,
        help="0 for a model trained without LiDAR intensity",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="runs/default/drive_demo")
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument(
        "--min-gt-points",
        type=int,
        default=10,
        help="annotated objects with fewer LiDAR points are not counted",
    )
    ap.add_argument("--lidar-only", action="store_true", help="classify without the camera")
    ap.add_argument(
        "--image-candidates",
        action="store_true",
        help="also take candidates from 2D person detections (frustum points)",
    )
    ap.add_argument(
        "--person-score", type=float, default=0.3, help="score threshold of the 2D detector"
    )
    ap.add_argument("--track", action="store_true", help="track objects and vote for the class")
    ap.add_argument(
        "--track-min-hits", type=int, default=2, help="frames before a track is published"
    )
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--snapshots", type=int, nargs="*", default=[20, 60, 100])
    args = ap.parse_args()
    summary = evaluate_drive(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
