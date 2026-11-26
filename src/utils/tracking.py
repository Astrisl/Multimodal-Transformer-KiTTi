"""Frame-to-frame association of classified objects with class voting.

A single frame decides the class from one random subset of points and one image patch,
so the label of an object flickers and clutter gets an object class now and then. The
tracker links the detections of consecutive frames (nearest centre, Hungarian
assignment) and replaces the per-frame probabilities by their mean over the last
frames of the track. A track is reported only after ``min_hits`` frames, which removes
labels that appear once and vanish.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    center: np.ndarray  # last position in the tracking frame
    velocity: np.ndarray  # per update step
    history: Deque[np.ndarray] = field(repr=False, default_factory=deque)
    hits: int = 1
    misses: int = 0

    def predicted(self) -> np.ndarray:
        return self.center + self.velocity * (self.misses + 1)


@dataclass(frozen=True)
class TrackedObject:
    """Tracker output for one detection of the current frame."""

    track_id: int
    probabilities: np.ndarray  # mean over the track's recent frames
    hits: int
    confirmed: bool


class ClassVotingTracker:
    """Tracks object centres and averages class probabilities along each track.

    Args:
        gate: Largest distance (m, ground plane) between a track's predicted position
            and a detection that may continue it.
        window: Number of recent frames whose probabilities are averaged.
        min_hits: Frames after which a track is confirmed.
        max_misses: Frames a track survives without a detection.
    """

    def __init__(
        self, gate: float = 2.0, window: int = 5, min_hits: int = 2, max_misses: int = 3
    ) -> None:
        self.gate = gate
        self.window = window
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(
        self, centers: np.ndarray, probabilities: np.ndarray, pose: Optional[np.ndarray] = None
    ) -> List[TrackedObject]:
        """Feeds the detections of one frame.

        Args:
            centers: Object centres [N, 3] in the sensor frame.
            probabilities: Class probabilities [N, C] of the same objects.
            pose: Sensor pose [4, 4] in a fixed frame; when given, tracking is done in
                that frame, so the motion of the vehicle does not move static objects.

        Returns:
            One :class:`TrackedObject` per detection, in the input order.
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        if pose is not None:
            centers = centers @ pose[:3, :3].T + pose[:3, 3]
        assigned = [-1] * len(centers)
        if self.tracks and len(centers):
            predicted = np.array([t.predicted()[:2] for t in self.tracks])
            cost = np.linalg.norm(predicted[:, None] - centers[None, :, :2], axis=2)
            rows, cols = linear_sum_assignment(np.where(cost <= self.gate, cost, 1e6))
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.gate:
                    assigned[c] = r

        matched = set()
        results: List[TrackedObject] = []
        new_tracks: List[Track] = []
        for i, (center, probs) in enumerate(zip(centers, probabilities)):
            if assigned[i] >= 0:
                track = self.tracks[assigned[i]]
                matched.add(assigned[i])
                track.velocity = (center - track.center) / (track.misses + 1)
                track.center, track.hits, track.misses = center, track.hits + 1, 0
            else:
                track = Track(self._next_id, center, np.zeros(3), deque(maxlen=self.window))
                self._next_id += 1
                new_tracks.append(track)
            track.history.append(np.asarray(probs, dtype=np.float64))
            results.append(
                TrackedObject(
                    track.track_id,
                    np.mean(track.history, axis=0),
                    track.hits,
                    track.hits >= self.min_hits,
                )
            )
        survivors = []
        for index, track in enumerate(self.tracks):
            if index not in matched:
                track.misses += 1
            if track.misses <= self.max_misses:
                survivors.append(track)
        self.tracks = survivors + new_tracks
        return results
