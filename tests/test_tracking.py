"""Tracker with class voting: stable labels, no one-frame false alarms."""

import numpy as np

from multimodal_transformer.utils.tracking import ClassVotingTracker

CAR = np.array([0.9, 0.05, 0.05])
PED = np.array([0.1, 0.8, 0.1])


def test_voting_removes_a_one_frame_flip():
    tracker = ClassVotingTracker(window=5, min_hits=2)
    center = np.array([[10.0, 2.0, 0.0]])
    labels = []
    for probs in (CAR, CAR, PED, CAR, CAR):
        (state,) = tracker.update(center, probs[None])
        labels.append(int(np.argmax(state.probabilities)))
    assert labels == [0, 0, 0, 0, 0]
    assert state.hits == 5 and state.confirmed


def test_track_is_confirmed_on_the_second_frame_and_ids_are_stable():
    tracker = ClassVotingTracker(min_hits=2)
    centers = np.array([[10.0, 2.0, 0.0], [20.0, -3.0, 0.0]])
    first = tracker.update(centers, np.stack([CAR, PED]))
    assert [s.confirmed for s in first] == [False, False]
    # Same objects in the other order, slightly moved.
    second = tracker.update(centers[::-1] + 0.2, np.stack([PED, CAR]))
    assert [s.confirmed for s in second] == [True, True]
    assert [s.track_id for s in second] == [first[1].track_id, first[0].track_id]
    # A detection far from every track starts a new, unconfirmed one.
    (third,) = tracker.update(np.array([[35.0, 10.0, 0.0]]), CAR[None])
    assert not third.confirmed and third.track_id not in {s.track_id for s in first}


def test_lost_track_is_dropped_and_empty_frames_are_fine():
    tracker = ClassVotingTracker(max_misses=1)
    tracker.update(np.array([[10.0, 0.0, 0.0]]), CAR[None])
    assert tracker.update(np.zeros((0, 3)), np.zeros((0, 3))) == []
    assert len(tracker.tracks) == 1
    tracker.update(np.zeros((0, 3)), np.zeros((0, 3)))
    assert tracker.tracks == []


def test_sensor_pose_keeps_static_objects_on_one_track():
    tracker = ClassVotingTracker(gate=1.0)
    ids = set()
    for step in range(4):  # the vehicle drives 3 m per frame towards a parked car
        pose = np.eye(4)
        pose[0, 3] = 3.0 * step
        (state,) = tracker.update(np.array([[30.0 - 3.0 * step, 2.0, 0.0]]), CAR[None], pose)
        ids.add(state.track_id)
    assert len(ids) == 1 and state.hits == 4


def test_moving_object_is_followed_by_its_velocity():
    tracker = ClassVotingTracker(gate=2.0)
    ids = set()
    for step in (0, 1, 2, 4, 5):  # 1.5 m per frame; the detection of frame 3 is missing
        if step == 4:
            tracker.update(np.zeros((0, 3)), np.zeros((0, 3)))
        (state,) = tracker.update(np.array([[10.0 + 1.5 * step, 0.0, 0.0]]), CAR[None])
        ids.add(state.track_id)
    # After the gap the object is 3 m from its last position - beyond the gate, but
    # where its velocity says it should be.
    assert len(ids) == 1 and state.hits == 5
