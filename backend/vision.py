from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .attention import Observation, Track

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - exposed as a health message at runtime
    mp = None


class FaceAnalyzer:
    def __init__(self, max_faces: int = 2) -> None:
        self.available = mp is not None
        self.error = ""
        self.mesh: Any = None
        self._legacy = False
        if self.available:
            try:
                if hasattr(mp, "solutions"):
                    self._legacy = True
                    self.mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=max_faces,
                        refine_landmarks=True,
                        min_detection_confidence=0.55,
                        min_tracking_confidence=0.55,
                    )
                else:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision

                    model_path = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
                    if not model_path.is_file():
                        raise FileNotFoundError("models/face_landmarker.task")
                    options = vision.FaceLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=max_faces,
                        min_face_detection_confidence=0.55,
                        min_tracking_confidence=0.55,
                    )
                    self.mesh = vision.FaceLandmarker.create_from_options(options)
            except Exception as exc:
                self.available = False
                self.error = f"Face Landmarker initialization failed: {type(exc).__name__}"
        else:
            self.error = "MediaPipe is not installed"
        self._last_timestamp_ms = 0

    @staticmethod
    def _point(landmarks: Any, index: int, width: int, height: int) -> tuple[float, float]:
        point = landmarks[index]
        return point.x * width, point.y * height

    def analyze(self, frame: np.ndarray) -> list[Observation]:
        if not self.available or self.mesh is None:
            return []
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._legacy:
            rgb.flags.writeable = False
            result = self.mesh.process(rgb)
            faces = [face.landmark for face in (result.multi_face_landmarks or [])]
        else:
            timestamp_ms = max(self._last_timestamp_ms + 1, int(time.monotonic() * 1000))
            self._last_timestamp_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.mesh.detect_for_video(image, timestamp_ms)
            faces = result.face_landmarks or []
        observations: list[Observation] = []
        for lm in faces[:2]:
            xs = [p.x * width for p in lm]
            ys = [p.y * height for p in lm]
            x1, x2 = max(0, int(min(xs))), min(width, int(max(xs)))
            y1, y2 = max(0, int(min(ys))), min(height, int(max(ys)))
            yaw, pitch = self._head_pose(lm, width, height)
            gaze_x, gaze_y = self._gaze(lm, width, height)
            observations.append(Observation((x1, y1, x2 - x1, y2 - y1), yaw, pitch, gaze_x, gaze_y))
        return observations

    def _head_pose(self, lm: Any, width: int, height: int) -> tuple[float, float]:
        indices = [1, 152, 33, 263, 61, 291]
        image_points = np.array([self._point(lm, i, width, height) for i in indices], dtype=np.float64)
        model_points = np.array(
            [(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26), (43.3, 32.7, -26),
             (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64
        )
        camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
        ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0
        matrix, _ = cv2.Rodrigues(rotation)
        angles = cv2.RQDecomp3x3(matrix)[0]
        def normalize(angle: float) -> float:
            if angle > 90.0:
                return angle - 180.0
            if angle < -90.0:
                return angle + 180.0
            return angle

        return normalize(float(angles[1])), normalize(float(angles[0]))

    def _gaze(self, lm: Any, width: int, height: int) -> tuple[float, float]:
        if len(lm) < 478:
            return 0.0, 0.0
        values = []
        for left, right, top, bottom, iris in [(33, 133, 159, 145, 468), (362, 263, 386, 374, 473)]:
            lx, _ = self._point(lm, left, width, height)
            rx, _ = self._point(lm, right, width, height)
            _, ty = self._point(lm, top, width, height)
            _, by = self._point(lm, bottom, width, height)
            ix, iy = self._point(lm, iris, width, height)
            cx, cy = (lx + rx) / 2, (ty + by) / 2
            values.append(((ix - cx) / max(abs(rx - lx), 1), (iy - cy) / max(abs(by - ty), 1)))
        return tuple(float(sum(v[i] for v in values) / len(values)) for i in range(2))  # type: ignore[return-value]


def anonymize_and_annotate(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Return a new frame. The source frame is never mutated or returned."""
    output = frame.copy()
    height, width = output.shape[:2]
    for track in tracks:
        x, y, w, h = track.box
        pad_x, pad_top, pad_bottom = int(w * 0.22), int(h * 0.28), int(h * 0.18)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_top)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_bottom)
        roi = output[y1:y2, x1:x2]
        if roi.size:
            small_w, small_h = max(6, (x2 - x1) // 18), max(6, (y2 - y1) // 18)
            pixels = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            output[y1:y2, x1:x2] = cv2.resize(pixels, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        color = (48, 205, 104) if track.attentive else (145, 155, 172)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = f"ID {track.id}  {'ATTENTIVE' if track.attentive else 'AWAY'}"
        cv2.rectangle(output, (x1, max(0, y1 - 34)), (min(width, x1 + 205), y1), color, -1)
        cv2.putText(output, label, (x1 + 8, max(22, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        cx, cy = x + w // 2, y + h // 2
        dx = int(max(-70, min(70, -(track.yaw * 2 + track.gaze_x * 80))))
        dy = int(max(-60, min(60, track.pitch * 2 + track.gaze_y * 60)))
        cv2.arrowedLine(output, (cx, cy), (cx + dx, cy + dy), color, 3, tipLength=.25)
    return output
