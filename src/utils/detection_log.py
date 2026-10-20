"""Relational journal of real-time detections (report section 5, Figure 2).

Schema::

    Session      (id PK, started_at, location)
    SensorFrame  (id PK, session_id FK, lidar_ts, camera_ts)
    Detection    (id PK, session_id FK, frame_id FK, timestamp, class_label,
                  confidence, bbox_obb)

``bbox_obb`` is stored as JSON ``{"center": [x, y, z], "size": [l, w, h], "yaw": a}``.
SQLite is used because it needs no server and works on the Jetson out of the box.
Writes go through a background thread so the ROS callback is never blocked by disk I/O.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Union

_SCHEMA = """
CREATE TABLE IF NOT EXISTS Session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    location    TEXT
);
CREATE TABLE IF NOT EXISTS SensorFrame (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES Session(id),
    lidar_ts    REAL NOT NULL,
    camera_ts   REAL
);
CREATE TABLE IF NOT EXISTS Detection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES Session(id),
    frame_id    INTEGER REFERENCES SensorFrame(id),
    timestamp   REAL NOT NULL,
    class_label TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox_obb    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detection_session ON Detection(session_id, timestamp);
"""


@dataclass(frozen=True)
class DetectionRecord:
    """One classified object, ready to be persisted."""

    class_label: str
    confidence: float
    center: Sequence[float]
    size: Sequence[float]
    yaw: float

    def obb_json(self) -> str:
        return json.dumps(
            {
                "center": [float(v) for v in self.center],
                "size": [float(v) for v in self.size],
                "yaw": float(self.yaw),
            }
        )


class DetectionJournal:
    """Synchronous SQLite repository for sessions, frames and detections."""

    def __init__(self, db_path: Union[str, Path], location: Optional[str] = None) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        started = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO Session(started_at, location) VALUES (?, ?)", (started, location)
        )
        self._conn.commit()
        self.session_id = int(cursor.lastrowid)

    def log_frame(
        self, lidar_ts: float, camera_ts: Optional[float], detections: Sequence[DetectionRecord]
    ) -> int:
        """Stores one synchronised frame and its detections in a single transaction."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO SensorFrame(session_id, lidar_ts, camera_ts) VALUES (?, ?, ?)",
                (self.session_id, lidar_ts, camera_ts),
            )
            frame_id = int(cursor.lastrowid)
            self._conn.executemany(
                "INSERT INTO Detection(session_id, frame_id, timestamp, class_label, "
                "confidence, bbox_obb) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        self.session_id,
                        frame_id,
                        lidar_ts,
                        det.class_label,
                        float(det.confidence),
                        det.obb_json(),
                    )
                    for det in detections
                ],
            )
        return frame_id

    def detections(self, session_id: Optional[int] = None) -> List[tuple]:
        """Returns ``(timestamp, class_label, confidence, bbox_obb)`` rows of a session."""
        sid = self.session_id if session_id is None else session_id
        return self._conn.execute(
            "SELECT timestamp, class_label, confidence, bbox_obb FROM Detection "
            "WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()

    def close(self) -> None:
        self._conn.close()


class AsyncDetectionJournal:
    """Non-blocking wrapper: frames are queued and written by a worker thread.

    If the queue is full (disk slower than the sensor) frames are dropped and counted
    rather than stalling the real-time pipeline.
    """

    def __init__(self, journal: DetectionJournal, max_queue: int = 256) -> None:
        self.journal = journal
        self.dropped_frames = 0
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self._worker = threading.Thread(target=self._run, name="detection-journal", daemon=True)
        self._worker.start()

    def submit(
        self, lidar_ts: float, camera_ts: Optional[float], detections: Sequence[DetectionRecord]
    ) -> None:
        try:
            self._queue.put_nowait((lidar_ts, camera_ts, list(detections)))
        except queue.Full:
            self.dropped_frames += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self.journal.log_frame(*item)

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        self._worker.join(timeout)
        self.journal.close()
