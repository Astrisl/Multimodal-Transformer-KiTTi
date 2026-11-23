"""Tests of the KITTI raw drive reader (synthetic drive on disk)."""

from pathlib import Path

import cv2
import numpy as np

from multimodal_transformer.data_pipeline.kitti_raw import (
    KittiRawDrive,
    oxts_to_poses,
    read_raw_calibration,
    read_tracklets,
)

CAM_TO_CAM = """calib_time: 09-Jan-2012 13:57:47
corner_dist: 9.950000e-02
R_rect_00: 1 0 0 0 1 0 0 0 1
P_rect_02: 721.5 0 609.5 44.8 0 721.5 172.8 0.2 0 0 1 0.003
"""
VELO_TO_CAM = """calib_time: 15-Mar-2012 11:37:16
R: 0 -1 0 0 0 -1 1 0 0
T: 0.0 -0.08 -0.27
delta_f: 0 0
"""
IMU_TO_VELO = """calib_time: 25-May-2012 16:47:16
R: 1 0 0 0 1 0 0 0 1
T: -0.8 0.3 -0.8
"""
TRACKLETS = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<boost_serialization signature="serialization::archive" version="9">
<tracklets class_id="0" tracking_level="0" version="0">
  <count>1</count>
  <item_version>1</item_version>
  <item class_id="1" tracking_level="0" version="1">
    <objectType>Car</objectType>
    <h>1.5</h><w>1.6</w><l>4.0</l>
    <first_frame>1</first_frame>
    <poses class_id="2" tracking_level="0" version="0">
      <count>2</count>
      <item_version>2</item_version>
      <item class_id="3" tracking_level="0" version="2">
        <tx>10.0</tx><ty>2.0</ty><tz>-1.7</tz><rx>0</rx><ry>0</ry><rz>0.5</rz>
      </item>
      <item><tx>11.0</tx><ty>2.0</ty><tz>-1.7</tz><rx>0</rx><ry>0</ry><rz>0.6</rz></item>
    </poses>
  </item>
</tracklets>
</boost_serialization>
"""


def make_drive(root: Path, frames: int = 3) -> Path:
    date = root / "2011_09_26"
    drive = date / "2011_09_26_drive_0001_sync"
    for sub in ("image_02/data", "velodyne_points/data", "oxts/data"):
        (drive / sub).mkdir(parents=True)
    (date / "calib_cam_to_cam.txt").write_text(CAM_TO_CAM)
    (date / "calib_velo_to_cam.txt").write_text(VELO_TO_CAM)
    (date / "calib_imu_to_velo.txt").write_text(IMU_TO_VELO)
    (drive / "tracklet_labels.xml").write_text(TRACKLETS)
    for i in range(frames):
        name = f"{i:010d}"
        np.random.rand(50, 4).astype(np.float32).tofile(
            drive / "velodyne_points/data" / f"{name}.bin"
        )
        cv2.imwrite(str(drive / "image_02/data" / f"{name}.png"), np.zeros((8, 8, 3), np.uint8))
        # Driving north-east: latitude grows, heading (yaw) 0.1 rad.
        oxts = [49.0 + i * 1e-5, 8.4, 110.0, 0.0, 0.0, 0.1] + [0.0] * 24
        (drive / "oxts/data" / f"{name}.txt").write_text(" ".join(map(str, oxts)))
    return drive


def test_calibration_matches_object_format(tmp_path):
    drive = make_drive(tmp_path)
    calib = read_raw_calibration(drive.parent)
    assert calib.P.shape == (3, 4) and calib.P[0, 0] == 721.5
    # A point 10 m ahead of the LiDAR is 10 m ahead (z) of the camera, slightly offset.
    rect = calib.lidar_to_rect(np.array([[10.0, 0.0, 0.0]]))
    np.testing.assert_allclose(rect[0], [0.0, -0.08, 9.73], atol=1e-6)


def test_oxts_poses_start_at_identity_and_follow_motion():
    oxts = np.array([[49.0, 8.4, 110.0, 0, 0, 0], [49.0001, 8.4, 110.0, 0, 0, 0]])
    poses = oxts_to_poses(oxts)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-9)
    # 1e-4 deg of latitude is ~11.1 m north (y) in the Mercator projection.
    assert 10.5 < poses[1, 1, 3] < 11.5
    assert abs(poses[1, 0, 3]) < 1e-6


def test_drive_reader(tmp_path):
    drive = KittiRawDrive(make_drive(tmp_path))
    assert len(drive) == 3
    assert drive.points(0).shape == (50, 4)
    assert drive.image_bgr(2).shape == (8, 8, 3)
    np.testing.assert_allclose(drive.lidar_poses[0], np.eye(4), atol=1e-9)
    assert drive.lidar_poses[2, :3, 3].any()


def test_tracklets(tmp_path):
    drive = make_drive(tmp_path)
    (track,) = read_tracklets(drive / "tracklet_labels.xml")
    assert track.object_type == "Car"
    np.testing.assert_allclose(track.size, [1.5, 1.6, 4.0])
    assert track.pose_at(0) is None
    position, yaw = track.pose_at(2)
    np.testing.assert_allclose(position, [11.0, 2.0, -1.7])
    assert yaw == 0.6
