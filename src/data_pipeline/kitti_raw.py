"""Reader for KITTI raw recordings (continuous drives), used to replay a real drive
through the ROS 2 pipeline.

Expected layout (as unpacked from the official ``*_sync.zip`` and ``*_calib.zip``)::

    <date>/calib_cam_to_cam.txt, calib_velo_to_cam.txt, calib_imu_to_velo.txt
    <date>/<date>_drive_XXXX_sync/image_02/data/0000000000.png
                                 /velodyne_points/data/0000000000.bin
                                 /oxts/data/0000000000.txt
                                 /tracklet_labels.xml            (optional)

The calibration is converted to the same ``Calibration`` (P2, R0_rect, Tr_velo_to_cam)
as KITTI Object, so the rest of the pipeline is unchanged. GPS/IMU (OXTS) records are
turned into LiDAR poses relative to the first frame, as in the KITTI raw devkit.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from multimodal_transformer.data_pipeline.kitti_reader import read_velodyne
from multimodal_transformer.utils.calibration import Calibration

PathLike = Union[str, Path]
EARTH_RADIUS = 6378137.0  # m, WGS84 equatorial radius used by the KITTI devkit


def read_calib_values(path: PathLike) -> Dict[str, np.ndarray]:
    """Parses a ``key: v1 v2 ...`` KITTI calibration file; non-numeric lines are skipped."""
    values: Dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()], dtype=np.float64)
        except ValueError:
            continue
    return values


def _rigid(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def read_raw_calibration(date_dir: PathLike, camera: int = 2) -> Calibration:
    """Builds the KITTI Object style calibration of colour camera ``camera`` (default 2)."""
    date_dir = Path(date_dir)
    cam = read_calib_values(date_dir / "calib_cam_to_cam.txt")
    velo = read_calib_values(date_dir / "calib_velo_to_cam.txt")
    tr = _rigid(velo["R"], velo["T"])[:3]
    return Calibration(P=cam[f"P_rect_0{camera}"], R0=cam["R_rect_00"], Tr_lidar_to_cam=tr)


def read_imu_to_velo(date_dir: PathLike) -> np.ndarray:
    """4x4 transform that maps IMU coordinates to Velodyne coordinates."""
    values = read_calib_values(Path(date_dir) / "calib_imu_to_velo.txt")
    return _rigid(values["R"], values["T"])


def oxts_to_poses(oxts: np.ndarray) -> np.ndarray:
    """IMU poses [N, 4, 4] relative to the first record (KITTI devkit convention).

    Args:
        oxts: OXTS records [N, >=6]: lat, lon (deg), alt (m), roll, pitch, yaw (rad).
    """
    oxts = np.atleast_2d(np.asarray(oxts, dtype=np.float64))
    scale = np.cos(np.deg2rad(oxts[0, 0]))
    poses = np.empty((len(oxts), 4, 4))
    for i, (lat, lon, alt, roll, pitch, yaw) in enumerate(oxts[:, :6]):
        t = np.array(
            [
                scale * np.deg2rad(lon) * EARTH_RADIUS,
                scale * EARTH_RADIUS * np.log(np.tan(np.deg2rad(90.0 + lat) / 2.0)),
                alt,
            ]
        )
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        poses[i] = _rigid(rz @ ry @ rx, t)
    return np.linalg.inv(poses[0]) @ poses


@dataclass
class Tracklet:
    """One annotated object track of a raw drive (box in the Velodyne frame)."""

    object_type: str
    size: np.ndarray  # h, w, l (m)
    first_frame: int
    translations: np.ndarray  # [T, 3] box bottom centre, Velodyne frame
    yaws: np.ndarray  # [T] rotation around z

    def pose_at(self, frame: int) -> Optional[tuple]:
        k = frame - self.first_frame
        if 0 <= k < len(self.translations):
            return self.translations[k], float(self.yaws[k])
        return None


def read_tracklets(path: PathLike) -> List[Tracklet]:
    """Parses ``tracklet_labels.xml`` of a raw drive."""
    root = ET.parse(str(path)).getroot()
    tracklets = []
    for item in root.find("tracklets").findall("item"):
        poses = item.find("poses").findall("item")
        tracklets.append(
            Tracklet(
                object_type=item.findtext("objectType"),
                size=np.array([float(item.findtext(k)) for k in ("h", "w", "l")]),
                first_frame=int(item.findtext("first_frame")),
                translations=np.array(
                    [[float(p.findtext(k)) for k in ("tx", "ty", "tz")] for p in poses]
                ).reshape(-1, 3),
                yaws=np.array([float(p.findtext("rz")) for p in poses]),
            )
        )
    return tracklets


class KittiRawDrive:
    """Frame accessor for one synchronised raw drive (``<date>_drive_XXXX_sync``)."""

    def __init__(self, drive_dir: PathLike, camera: int = 2) -> None:
        self.drive_dir = Path(drive_dir)
        self.date_dir = self.drive_dir.parent
        lidar_dir = self.drive_dir / "velodyne_points" / "data"
        if not lidar_dir.is_dir():
            raise FileNotFoundError(f"{lidar_dir} not found - is this a *_sync drive folder?")
        self.ids = sorted(p.stem for p in lidar_dir.glob("*.bin"))
        if not self.ids:
            raise FileNotFoundError(f"No scans in {lidar_dir}")
        self.image_dir = self.drive_dir / f"image_0{camera}" / "data"
        self.calibration = read_raw_calibration(self.date_dir, camera)
        self.lidar_poses = self._lidar_poses()

    def __len__(self) -> int:
        return len(self.ids)

    def points(self, index: int) -> np.ndarray:
        return read_velodyne(self.drive_dir / "velodyne_points" / "data" / f"{self.ids[index]}.bin")

    def image_bgr(self, index: int) -> np.ndarray:
        path = self.image_dir / f"{self.ids[index]}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return image

    def _lidar_poses(self) -> Optional[np.ndarray]:
        """LiDAR poses [N, 4, 4] relative to the first scan, or None without OXTS/IMU calib."""
        oxts_dir = self.drive_dir / "oxts" / "data"
        imu_calib = self.date_dir / "calib_imu_to_velo.txt"
        if not oxts_dir.is_dir() or not imu_calib.is_file():
            return None
        oxts = np.array([np.loadtxt(oxts_dir / f"{i}.txt")[:6] for i in self.ids])
        imu_poses = oxts_to_poses(oxts)
        velo_to_imu = np.linalg.inv(read_imu_to_velo(self.date_dir))
        world_velo = imu_poses @ velo_to_imu
        return np.linalg.inv(world_velo[0]) @ world_velo

    def tracklets(self) -> List[Tracklet]:
        path = self.drive_dir / "tracklet_labels.xml"
        return read_tracklets(path) if path.is_file() else []
