"""FR-01 (projection) and FR-02 (OBB derotation) tests."""

import numpy as np
import pytest

from multimodal_transformer.data_pipeline.image_cropper import box_to_image_roi
from multimodal_transformer.data_pipeline.kitti_reader import ObjectLabel
from multimodal_transformer.utils.calibration import Calibration
from multimodal_transformer.utils.geometry import OrientedBox, rotation_y, rotation_z


@pytest.mark.parametrize("ry", [0.0, 0.3, np.pi / 2, 2.5, -2.9, np.pi])
def test_kitti_obb_contains_points_for_any_rotation(ry):
    rng = np.random.default_rng(0)
    box = OrientedBox.from_kitti((1.0, 1.6, 10.0), (1.5, 1.8, 4.2), ry)
    local_inside = rng.uniform(-0.99, 0.99, (500, 3)) * box.half_extents
    local_outside = local_inside.copy()
    local_outside[:, 0] = np.sign(local_outside[:, 0] + 1e-9) * (box.half_extents[0] + 0.05)

    inside = local_inside @ box.rotation.T + box.center
    outside = local_outside @ box.rotation.T + box.center
    assert box.contains(inside).all()
    assert not box.contains(outside).any()


def test_kitti_box_center_is_half_height_above_bottom():
    box = OrientedBox.from_kitti((0.0, 2.0, 5.0), (1.6, 1.0, 3.0), 0.0)
    # Camera Y points down -> centre has smaller y than the bottom face.
    np.testing.assert_allclose(box.center, [0.0, 1.2, 5.0])
    np.testing.assert_allclose(box.half_extents, [1.5, 0.8, 0.5])


def test_corners_lie_on_box_boundary():
    box = OrientedBox.from_lidar((3, -1, 0.5), (4.0, 2.0, 1.5), 0.7)
    local = box.to_local(box.corners())
    np.testing.assert_allclose(np.abs(local), np.tile(box.half_extents, (8, 1)), atol=1e-9)


def test_rotations_are_orthonormal():
    for rot in (rotation_y(1.1), rotation_z(-2.3)):
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_lidar_rect_round_trip(kitti_calibration):
    pts = np.random.default_rng(1).uniform(-30, 30, (100, 3))
    back = kitti_calibration.rect_to_lidar(kitti_calibration.lidar_to_rect(pts))
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_projection_matches_kitti_devkit_formula(kitti_calibration):
    """Reference: y = P2 @ R0_rect @ Tr_velo_to_cam @ x (all homogeneous)."""
    c = kitti_calibration
    pts = np.array([[10.0, 1.0, -0.5], [25.0, -4.0, 0.3], [6.0, 0.0, -1.5]])
    r0 = np.eye(4)
    r0[:3, :3] = c.R0
    tr = np.vstack([c.Tr_lidar_to_cam, [0, 0, 0, 1]])
    homo = np.hstack([pts, np.ones((3, 1))])
    ref = (c.P @ r0 @ tr @ homo.T).T
    ref = ref[:, :2] / ref[:, 2:3]

    pixels, depth = c.lidar_to_image(pts)
    assert (depth > 0).all()
    # FR-01: projection error must be below 2 px; the implementation is exact.
    assert np.abs(pixels - ref).max() < 1e-6


def test_projected_3d_box_matches_annotated_2d_box(kitti_calibration):
    """FR-01 on real data: KITTI 000000 pedestrian, annotated 2D box 712.4,143,810.73,307.92."""
    line = "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01"
    label = ObjectLabel.from_line(line)
    roi = box_to_image_roi(label.oriented_box().corners(), kitti_calibration, (375, 1242), 0, 1)
    x1, y1, x2, y2 = roi
    ax1, ay1, ax2, ay2 = label.bbox_2d
    # The annotation is a tight box around the visible person; the projected 3D box also
    # covers the box depth, so horizontally it may only be wider (+ 3 px rounding).
    assert x1 <= ax1 + 3 and x2 >= ax2 - 3
    # Top and bottom edges are fixed by the object height: FR-01 requires < 2 px.
    assert abs(y1 - ay1) < 2 and abs(y2 - ay2) < 2


def test_box_behind_camera_has_no_image_roi(kitti_calibration):
    box = OrientedBox.from_kitti((0.0, 1.5, -5.0), (1.5, 1.6, 4.0), 0.0)
    assert box_to_image_roi(box.corners(), kitti_calibration, (375, 1242)) is None


def test_yaml_calibration(tmp_path, kitti_calibration):
    path = tmp_path / "calib.yaml"
    path.write_text(
        f"P: {kitti_calibration.P.flatten().tolist()}\n"
        f"R0: {kitti_calibration.R0.flatten().tolist()}\n"
        f"Tr_lidar_to_cam: {kitti_calibration.Tr_lidar_to_cam.flatten().tolist()}\n"
    )
    loaded = Calibration.from_yaml(path)
    np.testing.assert_allclose(loaded.P, kitti_calibration.P)
