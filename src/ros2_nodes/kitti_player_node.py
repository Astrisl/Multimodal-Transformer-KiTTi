"""Replays KITTI recordings as ROS 2 sensor topics, so the live pipeline can be tested and
demonstrated without hardware.

Two sources are supported:

* ``kitti_raw_dir`` - a continuous raw drive (``<date>/<date>_drive_XXXX_sync``). The
  car's motion from GPS/IMU is published as TF ``world_frame`` -> ``lidar_frame`` and as
  the travelled path, so in RViz2 (fixed frame ``world``) the car drives through the scene.
* ``kitti_split_dir`` - the independent frames of KITTI Object (``<kitti_root>/training``).

Publishes ``lidar_topic`` (PointCloud2, frame ``lidar_frame``) and ``image_topic``
(Image, bgr8) with identical stamps at ``rate_hz``. The calibration is written to
``calibration_out`` for the inference node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from multimodal_transformer.data_pipeline.kitti_raw import KittiRawDrive
from multimodal_transformer.data_pipeline.kitti_reader import KittiFrame, list_sample_ids
from multimodal_transformer.utils.calibration import Calibration

FIELDS = [
    PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, n in enumerate(("x", "y", "z", "intensity"))
]


def rotation_to_quaternion(r: np.ndarray) -> tuple:
    """(x, y, z, w) of a 3x3 rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


class KittiPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("kitti_player")
        self.declare_parameter("kitti_split_dir", "")
        self.declare_parameter("kitti_raw_dir", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("path_topic", "/ego/path")
        self.declare_parameter("lidar_frame", "lidar")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("loop", True)
        self.declare_parameter("calibration_out", "/tmp/kitti_calibration.yaml")

        raw_dir = self.get_parameter("kitti_raw_dir").value
        self.drive = KittiRawDrive(raw_dir) if raw_dir else None
        if self.drive is not None:
            self.count = len(self.drive)
            calibration = self.drive.calibration
            source = f"raw drive {self.drive.drive_dir.name}"
        else:
            self.split_dir = Path(self.get_parameter("kitti_split_dir").value)
            self.ids = list_sample_ids(self.split_dir)
            self.count = len(self.ids)
            calibration = KittiFrame(self.split_dir, self.ids[0]).calibration
            source = f"KITTI Object frames from {self.split_dir}"

        self.frame = self.get_parameter("lidar_frame").value
        self.world = self.get_parameter("world_frame").value
        self.loop = self.get_parameter("loop").value
        self.index = 0
        self.bridge = CvBridge()
        self.lidar_pub = self.create_publisher(
            PointCloud2, self.get_parameter("lidar_topic").value, 5
        )
        self.image_pub = self.create_publisher(Image, self.get_parameter("image_topic").value, 5)
        self.path_pub = self.create_publisher(PathMsg, self.get_parameter("path_topic").value, 1)
        self.tf_pub = TransformBroadcaster(self)
        self.path = PathMsg()
        self._write_calibration(calibration)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(f"Replaying {self.count} scans: {source}")

    def _write_calibration(self, calib: Calibration) -> None:
        data = {
            "P": calib.P.flatten().tolist(),
            "R0": calib.R0.flatten().tolist(),
            "Tr_lidar_to_cam": calib.Tr_lidar_to_cam.flatten().tolist(),
        }
        path = self.get_parameter("calibration_out").value
        Path(path).write_text(yaml.safe_dump(data), encoding="utf-8")
        self.get_logger().info(f"Calibration written to {path}")

    def _load(self, index: int) -> tuple:
        if self.drive is not None:
            return self.drive.points(index), self.drive.image_bgr(index)
        frame = KittiFrame(self.split_dir, self.ids[index])
        return frame.points, frame.image_bgr

    def _publish_pose(self, header: Header, index: int) -> None:
        if self.drive is None or self.drive.lidar_poses is None:
            return
        pose = self.drive.lidar_poses[index]
        qx, qy, qz, qw = rotation_to_quaternion(pose[:3, :3])
        tf = TransformStamped()
        tf.header = Header(stamp=header.stamp, frame_id=self.world)
        tf.child_frame_id = self.frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(v) for v in pose[:3, 3]
        )
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)

        if index == 0:
            self.path.poses.clear()
        stamped = PoseStamped()
        stamped.header = tf.header
        stamped.pose.position.x = tf.transform.translation.x
        stamped.pose.position.y = tf.transform.translation.y
        stamped.pose.position.z = tf.transform.translation.z
        stamped.pose.orientation = tf.transform.rotation
        self.path.header = tf.header
        self.path.poses.append(stamped)
        self.path_pub.publish(self.path)

    def tick(self) -> None:
        if self.index >= self.count:
            if not self.loop:
                return
            self.index = 0
        index = self.index
        self.index += 1
        points, image_bgr = self._load(index)

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        # TF first, so that RViz can place the scan as soon as it arrives.
        self._publish_pose(header, index)
        cloud = point_cloud2.create_cloud(header, FIELDS, points.astype(np.float32))
        image = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
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
