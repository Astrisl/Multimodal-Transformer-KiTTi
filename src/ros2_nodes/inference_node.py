"""Multimodal Transformer inference + publisher node (FR-04, FR-06).

Subscribes:
    ``lidar_topic``       sensor_msgs/PointCloud2       raw scan
    ``candidates_topic``  vision_msgs/Detection3DArray  OBBs from roi_extraction_node
    ``image_topic``       sensor_msgs/Image             RGB or IR camera
Publishes:
    ``detections_topic``  vision_msgs/Detection3DArray  boxes with class + score
    ``markers_topic``     visualization_msgs/MarkerArray  boxes + labels for RViz2

Scan and candidates carry the same stamp and are paired exactly. Camera frames arrive
at a different rate, so the latest frames are buffered and the one closest in time to
the scan is used. If it is further than ``max_sync_dt`` away (camera stalled, sync
lost) or too dark, the objects are classified from LiDAR alone instead of being dropped
(NFR-05), and the event is counted and reported.

Scan and candidate subscriptions keep only the newest message: when inference is slower
than the sensor, stale scans are skipped instead of queued, so latency stays bounded.
Camera frames are received in their own callback group on a multi-threaded executor, so
the image buffer keeps filling while a frame is being classified; otherwise the camera
queue overflows during inference and the frame matching the scan is lost.

Run::

    python3 -m multimodal_transformer.ros2_nodes.inference_node \\
        --ros-args --params-file config/ros2_pipeline.yaml
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi, crop_image
from multimodal_transformer.data_pipeline.lidar_cropper import crop_points_in_boxes
from multimodal_transformer.data_pipeline.normalization import PointCloudNormalizer
from multimodal_transformer.data_pipeline.sample_builder import (
    MultimodalSampleBuilder,
    build_image_transform,
    to_rgb,
)
from multimodal_transformer.model.inference_engine import ObjectClassifier, create_backend
from multimodal_transformer.ros2_nodes.conversions import (
    detection_to_box,
    detections_to_markers,
    make_detection,
    pointcloud2_to_array,
    stamp_to_sec,
)
from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.detection_log import (
    AsyncDetectionJournal,
    DetectionJournal,
    DetectionRecord,
)

PARAMETERS = {
    "lidar_topic": "/lidar/points",
    "image_topic": "/camera/image_raw",
    "candidates_topic": "/roi/candidates",
    "detections_topic": "/detected_objects",
    "markers_topic": "/detected_objects/markers",
    "calibration_file": "config/calibration_example.yaml",
    "use_camera": True,
    "max_sync_dt": 0.05,
    "image_buffer_size": 10,
    "sync_queue_size": 10,
    "image_padding": 5,
    "min_image_brightness": 15.0,
    "class_names": ["Car", "Pedestrian", "Cyclist"],
    "reject_classes": ["Background"],
    "num_points": 1024,
    "intensity_scale": 1.0,
    "min_points": 5,
    "min_confidence": 0.5,
    "backend": "torch",
    "checkpoint": "runs/default/best.pth",
    "onnx": "model.onnx",
    "engine": "model.engine",
    "device": "cuda",
    "fp16": True,
    "max_batch": 32,
    "journal_db": "",
    "session_location": "",
}


class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("multimodal_inference")
        for name, default in PARAMETERS.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in PARAMETERS}
        self.p = p

        self.class_names: List[str] = list(p["class_names"])
        self.calibration = Calibration.from_yaml(p["calibration_file"])
        self.builder = MultimodalSampleBuilder(
            normalizer=PointCloudNormalizer(p["num_points"], p["intensity_scale"]),
            image_transform=build_image_transform(),
            min_image_brightness=p["min_image_brightness"],
        )
        self.classifier = ObjectClassifier(
            create_backend(p, len(self.class_names)), self.class_names
        )
        self.bridge = CvBridge()

        self.journal: Optional[AsyncDetectionJournal] = None
        if p["journal_db"]:
            journal = DetectionJournal(p["journal_db"], p["session_location"] or None)
            self.journal = AsyncDetectionJournal(journal)
            self.get_logger().info(f"Journal session {journal.session_id} in {p['journal_db']}")

        self.det_pub = self.create_publisher(Detection3DArray, p["detections_topic"], 10)
        self.marker_pub = self.create_publisher(MarkerArray, p["markers_topic"], 10)

        self.images: Deque[Tuple[float, Image]] = deque(maxlen=p["image_buffer_size"])
        self.images_lock = threading.Lock()
        if p["use_camera"]:
            self.create_subscription(
                Image,
                p["image_topic"],
                self.on_image,
                qos_profile_sensor_data,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
        latest_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        scan_sub = message_filters.Subscriber(
            self, PointCloud2, p["lidar_topic"], qos_profile=latest_only
        )
        cand_sub = message_filters.Subscriber(
            self, Detection3DArray, p["candidates_topic"], qos_profile=latest_only
        )
        self.sync = message_filters.TimeSynchronizer(
            [scan_sub, cand_sub], queue_size=p["sync_queue_size"]
        )
        self.sync.registerCallback(self.on_frame)

        self.stats = {"frames": 0, "lidar_only_frames": 0, "errors": 0, "busy_s": 0.0}
        self.create_timer(5.0, self.report_stats)
        self.get_logger().info(f"Inference ready (backend={p['backend']})")

    # ---------------------------------------------------------------- callbacks
    def on_image(self, msg: Image) -> None:
        with self.images_lock:
            self.images.append((stamp_to_sec(msg.header.stamp), msg))

    def closest_image(self, stamp: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Camera frame closest to ``stamp`` as RGB, or (None, None) if out of sync."""
        with self.images_lock:
            buffered = list(self.images)
        if not buffered:
            return None, None
        cam_ts, msg = min(buffered, key=lambda item: abs(item[0] - stamp))
        if abs(cam_ts - stamp) > self.p["max_sync_dt"]:
            self.get_logger().warn(
                f"Camera out of sync by {abs(cam_ts - stamp) * 1000:.0f} ms, using LiDAR only",
                throttle_duration_sec=5.0,
            )
            return None, None
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:  # noqa: BLE001 - corrupt image -> LiDAR fallback
            self.get_logger().error(f"Image conversion failed: {exc}")
            return None, None
        return to_rgb(image, msg.encoding), cam_ts

    def on_frame(self, scan_msg: PointCloud2, candidates: Detection3DArray) -> None:
        start = time.perf_counter()
        lidar_ts = stamp_to_sec(scan_msg.header.stamp)
        output = Detection3DArray(header=candidates.header)
        try:
            if candidates.detections:
                output.detections, records, camera_ts = self.classify(
                    scan_msg, candidates, lidar_ts
                )
                if self.journal is not None:
                    self.journal.submit(lidar_ts, camera_ts, records)
        except Exception as exc:  # noqa: BLE001 - inference failure must not stop the node
            self.stats["errors"] += 1
            self.get_logger().error(f"Inference failed: {exc}", throttle_duration_sec=2.0)
            return
        self.det_pub.publish(output)
        self.marker_pub.publish(detections_to_markers(output))
        self.stats["frames"] += 1
        self.stats["busy_s"] += time.perf_counter() - start

    # ------------------------------------------------------------------- logic
    def classify(self, scan_msg: PointCloud2, candidates: Detection3DArray, lidar_ts: float):
        scan = pointcloud2_to_array(scan_msg)
        image, camera_ts = self.closest_image(lidar_ts) if self.p["use_camera"] else (None, None)
        if image is None:
            self.stats["lidar_only_frames"] += 1

        boxes, patches, clouds = [], [], []
        parsed = [detection_to_box(det) for det in candidates.detections]
        crops = crop_points_in_boxes(scan, [box for box, _, _ in parsed])
        for (box, yaw, size), points in zip(parsed, crops):
            if points.shape[0] < self.p["min_points"]:
                continue
            patch = None
            if image is not None:
                corners_rect = self.calibration.lidar_to_rect(box.corners())
                roi = box_to_image_roi(
                    corners_rect, self.calibration, image.shape[:2], self.p["image_padding"]
                )
                patch = crop_image(image, roi) if roi else None
            boxes.append((box, yaw, size))
            patches.append(patch)
            clouds.append(points)

        predictions = []
        step = self.p["max_batch"]
        for i in range(0, len(clouds), step):
            batch = self.builder.build_batch(patches[i : i + step], clouds[i : i + step])
            predictions += self.classifier.predict(batch)

        detections, records = [], []
        for (box, yaw, size), pred in zip(boxes, predictions):
            if pred.confidence < self.p["min_confidence"]:
                continue
            if pred.class_name in self.p["reject_classes"]:  # clutter, not an object
                continue
            detections.append(
                make_detection(
                    candidates.header, box.center, size, yaw, pred.class_name, pred.confidence,
                    det_id=str(len(detections)),
                )  # fmt: skip
            )
            records.append(DetectionRecord(pred.class_name, pred.confidence, box.center, size, yaw))
        return detections, records, camera_ts

    def report_stats(self) -> None:
        s = self.stats
        if s["frames"] == 0:
            return
        ms = 1000 * s["busy_s"] / s["frames"]
        dropped = self.journal.dropped_frames if self.journal else 0
        self.get_logger().info(
            f"{s['frames'] / 5.0:.1f} FPS in, {ms:.1f} ms/frame (max {1000 / ms:.0f} FPS), "
            f"LiDAR-only {s['lidar_only_frames']}, errors {s['errors']}, "
            f"journal dropped {dropped}"
        )
        self.stats = {key: 0 if key != "busy_s" else 0.0 for key in s}

    def destroy_node(self) -> None:
        if self.journal is not None:
            self.journal.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = InferenceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
