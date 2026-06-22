from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable


@dataclass
class Observation:
    box: tuple[int, int, int, int]
    yaw: float
    pitch: float
    gaze_x: float
    gaze_y: float


@dataclass
class Calibration:
    active: bool = False
    started_at: float = 0.0
    duration: float = 3.0
    samples: list[tuple[float, float, float, float]] = field(default_factory=list)
    center: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    calibrated: bool = False

    def start(self, now: float | None = None) -> None:
        self.active = True
        self.started_at = monotonic() if now is None else now
        self.samples.clear()

    def add(self, values: tuple[float, float, float, float], now: float | None = None) -> None:
        if not self.active:
            return
        current = monotonic() if now is None else now
        self.samples.append(values)
        self.tick(current)

    def tick(self, now: float | None = None) -> None:
        if not self.active:
            return
        current = monotonic() if now is None else now
        if current - self.started_at >= self.duration:
            if self.samples:
                cols = tuple(zip(*self.samples))
                self.center = tuple(sum(col) / len(col) for col in cols)  # type: ignore[assignment]
                self.calibrated = True
            self.active = False

    def skip(self) -> None:
        self.active = False
        self.samples.clear()
        self.center = (0.0, 0.0, 0.0, 0.0)
        self.calibrated = False

    def progress(self, now: float | None = None) -> float:
        if not self.active:
            return 1.0 if self.calibrated else 0.0
        current = monotonic() if now is None else now
        return min(1.0, max(0.0, (current - self.started_at) / self.duration))

    def is_attentive(self, obs: Observation) -> bool:
        cy, cp, cgx, cgy = self.center
        # Broad defaults are intentional: this is a TV-direction estimate, not eye tracking.
        return (
            abs(obs.yaw - cy) <= 18.0
            and abs(obs.pitch - cp) <= 15.0
            and abs(obs.gaze_x - cgx) <= 0.22
            and abs(obs.gaze_y - cgy) <= 0.24
        )


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    attentive: bool = False
    candidate_since: float = 0.0
    last_seen: float = 0.0
    attentive_seconds: float = 0.0
    last_tick: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0


class AttentionTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = box
        return x + w / 2, y + h / 2

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(
        self, observations: Iterable[Observation], calibration: Calibration, now: float | None = None
    ) -> list[Track]:
        current = monotonic() if now is None else now
        unmatched = set(self.tracks)
        result: list[Track] = []

        for obs in observations:
            ox, oy = self._center(obs.box)
            best_id = None
            best_distance = float("inf")
            for track_id in unmatched:
                tx, ty = self._center(self.tracks[track_id].box)
                distance = ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5
                max_distance = max(obs.box[2], obs.box[3]) * 0.8
                if distance < best_distance and distance < max_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = Track(best_id, obs.box, last_seen=current, last_tick=current)
            else:
                unmatched.remove(best_id)

            track = self.tracks[best_id]
            if track.attentive:
                track.attentive_seconds += max(0.0, current - track.last_tick)
            desired = calibration.is_attentive(obs)
            if desired != track.attentive:
                if track.candidate_since == 0.0:
                    track.candidate_since = current
                threshold = 0.5 if desired else 0.8
                if current - track.candidate_since >= threshold:
                    track.attentive = desired
                    track.candidate_since = 0.0
            else:
                track.candidate_since = 0.0

            track.box = obs.box
            track.last_seen = current
            track.last_tick = current
            track.yaw, track.pitch = obs.yaw, obs.pitch
            track.gaze_x, track.gaze_y = obs.gaze_x, obs.gaze_y
            result.append(track)

        for track_id in list(self.tracks):
            if current - self.tracks[track_id].last_seen > 1.0:
                del self.tracks[track_id]
        return result
