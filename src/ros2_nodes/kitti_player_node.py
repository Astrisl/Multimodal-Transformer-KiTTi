"""Replays KITTI frames as ROS 2 sensor topics, so the live pipeline can be tested and
demonstrated without hardware.

Publishes ``lidar_topic`` (PointCloud2, frame ``lidar_frame``) and ``image_topic``
(Image, bgr8) with identical stamps at ``rate_hz``. The calibration of the first frame
is written to ``calibration_out`` for the inference node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, list_sample_ids

FIELDS = [
    PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, n in enumerate(("x", "y", "z", "intensity"))
]


class KittiPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("kitti_player")
        self.declare_parameter("kitti_split_dir", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("lidar_frame", "lidar")
        self.declare_parameter("loop", True)
        self.declare_parameter("calibration_out", "/tmp/kitti_calibration.yaml")

        self.split_dir = Path(self.get_parameter("kitti_split_dir").value)
        self.ids = list_sample_ids(self.split_dir)
        self.frame = self.get_parameter("lidar_frame").value
        self.loop = self.get_parameter("loop").value
        self.index = 0
        self.bridge = CvBridge()
        self.lidar_pub = self.create_publisher(
            PointCloud2, self.get_parameter("lidar_topic").value, 5
        )
        self.image_pub = self.create_publisher(Image, self.get_parameter("image_topic").value, 5)
        self._write_calibration(KittiFrame(self.split_dir, self.ids[0]))
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(f"Replaying {len(self.ids)} KITTI frames from {self.split_dir}")

    def _write_calibration(self, frame: KittiFrame) -> None:
        calib = frame.calibration
        data = {
            "P": calib.P.flatten().tolist(),
            "R0": calib.R0.flatten().tolist(),
            "Tr_lidar_to_cam": calib.Tr_lidar_to_cam.flatten().tolist(),
        }
        path = self.get_parameter("calibration_out").value
        Path(path).write_text(yaml.safe_dump(data), encoding="utf-8")
        self.get_logger().info(f"Calibration written to {path}")

    def tick(self) -> None:
        if self.index >= len(self.ids):
            if not self.loop:
                return
            self.index = 0
        frame = KittiFrame(self.split_dir, self.ids[self.index])
        self.index += 1

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        cloud = point_cloud2.create_cloud(header, FIELDS, frame.points.astype(np.float32))
        image = self.bridge.cv2_to_imgmsg(frame.image_bgr, encoding="bgr8")
        image.header = Header(stamp=header.stamp, frame_id="camera")
        self.lidar_pub.publish(cloud)
        self.image_pub.publish(image)


def main() -> None:
    rclpy.init()
    node = KittiPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
