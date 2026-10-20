"""ROI Extraction node: LiDAR scan -> object candidates with oriented boxes.

Subscribes:  ``lidar_topic``        sensor_msgs/PointCloud2
Publishes:   ``candidates_topic``   vision_msgs/Detection3DArray (boxes, no class yet)

The header (stamp + frame) of every output equals the input scan, which lets the
inference node pair candidates with their scan exactly.

Run::

    python3 -m multimodal_transformer.ros2_nodes.roi_extraction_node \\
        --ros-args --params-file config/ros2_pipeline.yaml
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

from multimodal_transformer.data_pipeline.roi_extraction import RoiConfig, RoiExtractor
from multimodal_transformer.ros2_nodes.conversions import make_detection, pointcloud2_to_array


class RoiExtractionNode(Node):
    def __init__(self) -> None:
        super().__init__("roi_extraction")
        defaults = RoiConfig()
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("candidates_topic", "/roi/candidates")
        for name, value in vars(defaults).items():
            self.declare_parameter(name, list(value) if isinstance(value, tuple) else value)

        values = {}
        for name, default in vars(defaults).items():
            value = self.get_parameter(name).value
            values[name] = tuple(value) if isinstance(default, tuple) else value
        cfg = RoiConfig(**values)
        self.extractor = RoiExtractor(cfg)
        self.publisher = self.create_publisher(
            Detection3DArray, self.get_parameter("candidates_topic").value, 10
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            self.on_scan,
            qos_profile_sensor_data,
        )
        self._times = []
        self.get_logger().info(f"ROI extraction ready: {cfg}")

    def on_scan(self, msg: PointCloud2) -> None:
        start = time.perf_counter()
        try:
            scan = pointcloud2_to_array(msg)
            candidates = self.extractor.extract(scan)
        except Exception as exc:  # noqa: BLE001 - a bad scan must not kill the node
            self.get_logger().error(f"ROI extraction failed: {exc}")
            return

        out = Detection3DArray(header=msg.header)
        for idx, cand in enumerate(candidates):
            out.detections.append(
                make_detection(
                    msg.header, cand.box.center, cand.size_lwh, cand.yaw, det_id=str(idx)
                )
            )
        self.publisher.publish(out)

        self._times.append(time.perf_counter() - start)
        if len(self._times) >= 100:
            mean_ms = 1000 * sum(self._times) / len(self._times)
            self.get_logger().info(f"ROI extraction: {mean_ms:.1f} ms/scan (mean of 100)")
            self._times.clear()


def main() -> None:
    rclpy.init()
    node = RoiExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
