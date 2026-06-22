from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from .attention import AttentionTracker, Calibration
from .vision import FaceAnalyzer, anonymize_and_annotate


class VisionEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._source: int | str = 0
        self._source_label = "Camera 0"
        self._status = "starting"
        self._fps = 0.0
        self._started_at = time.monotonic()
        self._analyzer = FaceAnalyzer(max_faces=2)
        self._tracker = AttentionTracker()
        self.calibration = Calibration()
        self._timeline: deque[dict[str, Any]] = deque(maxlen=180)
        self._last_timeline_second = -1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="privacy-vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._jpeg = None
            self._timeline.clear()
        self._tracker.reset()

    def set_source(self, kind: str, value: str | int | None = None) -> None:
        if kind == "camera":
            index = int(value or 0)
            source: int | str = index
            label = f"Camera {index}"
        elif kind == "video":
            path = Path(str(value or "")).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                raise ValueError("Video path does not exist or has an unsupported extension")
            source, label = str(path), path.name
        else:
            raise ValueError("Source kind must be camera or video")
        with self._lock:
            self._source = source
            self._source_label = label
            self._status = "switching"
        self._tracker.reset()

    def start_calibration(self) -> None:
        self.calibration.start()

    def skip_calibration(self) -> None:
        self.calibration.skip()

    def reset_session(self) -> None:
        self._tracker.reset()
        self.calibration.skip()
        with self._lock:
            self._timeline.clear()
            self._started_at = time.monotonic()

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        self.calibration.tick(now)
        tracks = sorted(self._tracker.tracks.values(), key=lambda item: item.id)
        visible = [track for track in tracks if now - track.last_seen <= 1.0]
        attentive_count = sum(track.attentive for track in visible)
        if not visible:
            overall = "no_people"
        elif attentive_count == len(visible):
            overall = "all"
        elif attentive_count:
            overall = "partial"
        else:
            overall = "none"
        with self._lock:
            return {
                "status": self._status,
                "modelAvailable": self._analyzer.available,
                "modelMessage": self._analyzer.error,
                "source": self._source_label,
                "fps": round(self._fps, 1),
                "sessionSeconds": round(now - self._started_at),
                "people": len(visible),
                "attentivePeople": attentive_count,
                "averageAttentionSeconds": round(
                    sum(item.attentive_seconds for item in visible) / len(visible), 1
                ) if visible else 0,
                "overall": overall,
                "calibration": {
                    "active": self.calibration.active,
                    "calibrated": self.calibration.calibrated,
                    "progress": round(self.calibration.progress(), 3),
                },
                "tracks": [
                    {
                        "id": item.id,
                        "attentive": item.attentive,
                        "attentionSeconds": round(item.attentive_seconds, 1),
                        "yaw": round(item.yaw, 1),
                        "pitch": round(item.pitch, 1),
                    }
                    for item in visible
                ],
                "timeline": list(self._timeline),
            }

    def _run(self) -> None:
        capture = None
        active_source: int | str | None = None
        frame_counter, fps_started = 0, time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                source = self._source
            if capture is None or source != active_source:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
                active_source = source
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                with self._lock:
                    self._status = "running" if capture.isOpened() else "source_error"

            ok, frame = capture.read() if capture is not None else (False, None)
            if not ok or frame is None:
                if isinstance(active_source, str) and capture is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue

            observations = self._analyzer.analyze(frame)
            now = time.monotonic()
            if self.calibration.active and observations:
                obs = observations[0]
                self.calibration.add((obs.yaw, obs.pitch, obs.gaze_x, obs.gaze_y), now)
            tracks = self._tracker.update(observations, self.calibration, now)
            safe_frame = anonymize_and_annotate(frame, tracks)
            del frame  # Keep the raw frame's lifetime inside this loop only.
            encoded, buffer = cv2.imencode(".jpg", safe_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if encoded:
                with self._lock:
                    self._jpeg = buffer.tobytes()

            frame_counter += 1
            elapsed = now - fps_started
            if elapsed >= 1.0:
                with self._lock:
                    self._fps = frame_counter / elapsed
                frame_counter, fps_started = 0, now
            second = int(now - self._started_at)
            if second != self._last_timeline_second:
                self._last_timeline_second = second
                visible = [t for t in tracks if now - t.last_seen <= 1.0]
                self._timeline.append({
                    "second": second,
                    "state": "all" if visible and all(t.attentive for t in visible)
                    else "partial" if any(t.attentive for t in visible) else "none",
                })
        if capture is not None:
            capture.release()
