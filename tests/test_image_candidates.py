"""Image-guided candidates: a person next to a wall, which clustering alone merges."""

import numpy as np

from multimodal_transformer.data_pipeline.image_candidates import (
    densest_range,
    duplicate_losers,
    frustum_candidates,
)
from multimodal_transformer.data_pipeline.roi_extraction import (
    ObjectCandidate,
    RoiConfig,
    RoiExtractor,
)
from multimodal_transformer.utils.geometry import OrientedBox


def person_by_wall(rng):
    ground = np.column_stack(
        [rng.uniform(0, 40, 20000), rng.uniform(-20, 20, 20000), rng.normal(-1.7, 0.02, 20000)]
    )
    person = OrientedBox.from_lidar((10.0, 0.0, -0.8), (0.5, 0.6, 1.8), 0.0)
    body = rng.uniform(-1, 1, (600, 3)) * person.half_extents + person.center
    # A wall 0.25 m behind the person's back: closer than the cluster tolerance (0.5 m).
    wall = np.column_stack(
        [rng.normal(10.5, 0.01, 6000), rng.uniform(-2, 2, 6000), rng.uniform(-1.7, 1.0, 6000)]
    )
    # A LiDAR sees one surface per ray: the part of the wall behind the person is hidden.
    wall = wall[(np.abs(wall[:, 1]) > 0.35) | (wall[:, 2] > 0.15)]
    xyz = np.vstack([ground, body, wall])
    return np.column_stack([xyz, rng.uniform(0, 1, len(xyz))]), person


def box_in_image(box, calibration):
    pixels, _ = calibration.lidar_to_image(box.corners())
    return np.array([[*pixels.min(axis=0), *pixels.max(axis=0), 0.9]])


def test_person_merged_with_wall_is_found_through_the_image(kitti_calibration):
    scan, person = person_by_wall(np.random.default_rng(0))
    scene = RoiExtractor(RoiConfig(), seed=0).analyse(scan)
    near = [c for c in scene.candidates if abs(c.box.center[0] - 10.0) < 1.5]
    assert near and all(max(c.size_lwh[:2]) > 3.0 for c in near)  # person + wall, one cluster

    found = frustum_candidates(scene, box_in_image(person, kitti_calibration), kitti_calibration)
    assert len(found) == 1 and found[0].source == "image"
    np.testing.assert_allclose(found[0].box.center[:2], person.center[:2], atol=0.15)
    assert max(found[0].size_lwh[:2]) < 1.0
    np.testing.assert_allclose(found[0].size_lwh[2], 1.8, atol=0.15)  # reaches the ground
    # The wall is excluded, the person's points are kept at full resolution.
    assert 550 < found[0].points.shape[0] < 750


def test_no_candidate_without_points_or_detections(kitti_calibration):
    scan, person = person_by_wall(np.random.default_rng(1))
    scene = RoiExtractor(RoiConfig(), seed=0).analyse(scan)
    assert frustum_candidates(scene, np.zeros((0, 5)), kitti_calibration) == []
    sky = np.array([[600.0, 0.0, 640.0, 40.0, 0.9]])  # a 2D box with no LiDAR points
    assert frustum_candidates(scene, sky, kitti_calibration) == []
    twice = np.vstack([box_in_image(person, kitti_calibration)] * 2)  # one person, two boxes
    assert len(frustum_candidates(scene, twice, kitti_calibration)) == 1


def test_densest_range_prefers_the_nearest_of_equal_intervals():
    ranges = np.array([5.0, 5.2, 5.4, 9.0, 9.1, 9.2, 20.0])
    assert densest_range(ranges, 1.0) == (5.0, 3)
    assert densest_range(np.array([9.0, 9.1, 9.2, 9.3, 5.0]), 1.0) == (9.0, 4)


def test_cluster_with_object_class_wins_over_image_candidate():
    def cand(x, size, source):
        box = OrientedBox.from_lidar((x, 0.0, 0.0), size, 0.0)
        return ObjectCandidate(box, 0.0, np.array(size), np.zeros((0, 4)), source)

    cyclist = cand(10.0, (1.8, 0.6, 1.7), "cluster")
    rider = cand(10.2, (0.5, 0.6, 1.7), "image")
    wall = cand(20.0, (6.0, 0.4, 2.0), "cluster")
    person = cand(20.3, (0.5, 0.6, 1.8), "image")
    far = cand(30.0, (0.5, 0.6, 1.8), "image")
    candidates = [cyclist, rider, wall, person, far]
    names = ["Cyclist", "Pedestrian", "Background", "Pedestrian", "Pedestrian"]
    # The rider duplicates the cyclist's cluster; the wall cluster hides a person.
    assert duplicate_losers(candidates, names) == [1, 2]
    names[3] = "Background"  # nothing found in that 2D box after all
    assert duplicate_losers(candidates, names) == [1, 3]
