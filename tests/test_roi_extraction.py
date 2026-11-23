"""Live ROI extraction on a synthetic scene: ground plane + objects."""

import numpy as np

from multimodal_transformer.data_pipeline.roi_extraction import RoiConfig, RoiExtractor
from multimodal_transformer.utils.geometry import OrientedBox


def make_scene(rng):
    ground = np.column_stack(
        [rng.uniform(0, 40, 20000), rng.uniform(-20, 20, 20000), rng.normal(-1.7, 0.02, 20000)]
    )
    car = OrientedBox.from_lidar((12.0, 3.0, -0.95), (4.2, 1.8, 1.5), 0.5)
    person = OrientedBox.from_lidar((8.0, -4.0, -0.8), (0.6, 0.5, 1.8), 0.0)
    objects = []
    for box, n in ((car, 3000), (person, 600)):
        local = rng.uniform(-1, 1, (n, 3)) * box.half_extents
        objects.append(local @ box.rotation.T + box.center)
    xyz = np.vstack([ground, *objects])
    return np.column_stack([xyz, rng.uniform(0, 1, len(xyz))]), car, person


def test_extracts_objects_and_removes_ground():
    scan, car, person = make_scene(np.random.default_rng(0))
    candidates = RoiExtractor(RoiConfig(), seed=0).extract(scan)
    assert len(candidates) == 2

    by_distance = sorted(candidates, key=lambda c: np.linalg.norm(c.box.center[:2]))
    found_person, found_car = by_distance
    np.testing.assert_allclose(found_car.box.center[:2], car.center[:2], atol=0.2)
    np.testing.assert_allclose(sorted(found_car.size_lwh[:2]), [1.8, 4.2], atol=0.25)
    np.testing.assert_allclose(found_car.size_lwh[2], 1.5, atol=0.15)
    # Heading is recovered up to the pi ambiguity of PCA.
    assert abs(np.sin(found_car.yaw - 0.5)) < 0.1
    np.testing.assert_allclose(found_person.box.center[:2], person.center[:2], atol=0.2)
    # Full-resolution points (not voxelised) are returned for classification.
    assert found_car.points.shape[0] > 2500


def test_empty_scan_gives_no_candidates():
    assert RoiExtractor().extract(np.zeros((0, 4))) == []


def test_crop_region_limits_azimuth():
    points = np.array([[10.0, 0.0, 0.0, 0], [10.0, 9.0, 0.0, 0], [5.0, 10.0, 0.0, 0]])
    assert len(RoiExtractor(RoiConfig()).crop_region(points)) == 3
    kept = RoiExtractor(RoiConfig(max_azimuth_deg=45.0)).crop_region(points)
    np.testing.assert_allclose(kept[:, :2], [[10.0, 0.0], [10.0, 9.0]])
