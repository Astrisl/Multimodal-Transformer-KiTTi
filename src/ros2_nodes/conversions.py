"""Conversions between ROS 2 messages and the NumPy structures of the pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import BoundingBox3D, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from multimodal_transformer.utils.geometry import OrientedBox, box_edges

# Different drivers name the return strength differently (Velodyne / Ouster / Livox).
INTENSITY_FIELDS = ("intensity", "reflectivity", "i")

CLASS_COLORS = {
    "Car": (0.1, 0.6, 1.0),
    "Pedestrian": (1.0, 0.3, 0.2),
    "Cyclist": (1.0, 0.9, 0.1),
}
UNKNOWN_COLOR = (0.7, 0.7, 0.7)


def stamp_to_sec(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    """Converts a cloud to float32 [N, 4] (x, y, z, intensity); NaN points are removed.

    Clouds without an intensity channel get intensity 0.
    """
    names = {f.name for f in msg.fields}
    intensity = next((n for n in INTENSITY_FIELDS if n in names), None)
    fields = ["x", "y", "z"] + ([intensity] if intensity else [])
    raw = point_cloud2.read_points_numpy(msg, field_names=fields, skip_nans=True)
    out = np.zeros((raw.shape[0], 4), dtype=np.float32)
    out[:, : raw.shape[1]] = raw
    return out


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))


def quaternion_to_yaw(q: Quaternion) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def make_detection(
    header: Header,
    center: Sequence[float],
    size_lwh: Sequence[float],
    yaw: float,
    class_name: Optional[str] = None,
    score: float = 0.0,
    det_id: str = "",
) -> Detection3D:
    det = Detection3D(header=header, id=det_id)
    det.bbox = BoundingBox3D(
        center=Pose(
            position=Point(x=float(center[0]), y=float(center[1]), z=float(center[2])),
            orientation=yaw_to_quaternion(yaw),
        ),
        size=Vector3(x=float(size_lwh[0]), y=float(size_lwh[1]), z=float(size_lwh[2])),
    )
    if class_name is not None:
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_name
        hyp.hypothesis.score = float(score)
        hyp.pose.pose = det.bbox.center
        det.results.append(hyp)
    return det


def detection_to_box(det: Detection3D) -> Tuple[OrientedBox, float, np.ndarray]:
    """Inverse of :func:`make_detection` for the box part."""
    p, s = det.bbox.center.position, det.bbox.size
    yaw = quaternion_to_yaw(det.bbox.center.orientation)
    size = np.array([s.x, s.y, s.z])
    return OrientedBox.from_lidar((p.x, p.y, p.z), size, yaw), yaw, size


def detections_to_markers(array: Detection3DArray, lifetime_sec: float = 0.2) -> MarkerArray:
    """Box edges + text label per detection, for RViz2 (FR-06)."""
    markers = MarkerArray()
    markers.markers.append(Marker(header=array.header, action=Marker.DELETEALL))
    lifetime = Duration(sec=int(lifetime_sec), nanosec=int((lifetime_sec % 1) * 1e9))
    for idx, det in enumerate(array.detections):
        box, _, _ = detection_to_box(det)
        label = det.results[0].hypothesis.class_id if det.results else "?"
        score = det.results[0].hypothesis.score if det.results else 0.0
        r, g, b = CLASS_COLORS.get(label, UNKNOWN_COLOR)
        color = ColorRGBA(r=r, g=g, b=b, a=1.0)

        lines = Marker(header=array.header, ns="boxes", id=idx, type=Marker.LINE_LIST)
        lines.scale.x = 0.05
        lines.color = color
        lines.lifetime = lifetime
        lines.pose.orientation.w = 1.0
        corners = box.corners()
        lines.points = [
            Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            for i, j in box_edges()
            for c in (corners[i], corners[j])
        ]

        text = Marker(header=array.header, ns="labels", id=idx, type=Marker.TEXT_VIEW_FACING)
        text.text = f"{label} {score:.2f}"
        text.scale.z = 0.4
        text.color = color
        text.lifetime = lifetime
        text.pose.position = Point(
            x=float(box.center[0]), y=float(box.center[1]),
            z=float(box.center[2] + box.half_extents[2] + 0.3),
        )  # fmt: skip
        text.pose.orientation.w = 1.0
        markers.markers += [lines, text]
    return markers


def as_float_list(values: Iterable) -> List[float]:
    return [float(v) for v in values]
