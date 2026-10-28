"""Iris/pupil detector for one goggle camera frame.

Searches an aim box around the eye, prefers candidates near the last lock and takes the
darkness-weighted centre so the pupil dominates.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

AIM_BOX_WIDTH = 0.70   # fraction of frame width searched around the aim point
AIM_BOX_HEIGHT = 0.42  # fraction of frame height
FULL_FRAME_MARGIN = 0.04
DARK_PERCENTILES = (3, 6, 10)
MIN_AREA_FRACTION = 0.004
MAX_AREA_FRACTION = 0.16
MIN_LOCK_SCORE = 0.5
PREVIOUS_RADIUS = 0.18  # normalized distance where the continuity bonus fades out
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
RING_KERNEL = np.ones((17, 17), dtype=np.uint8)
_LOCAL = threading.local()  # OpenCV CLAHE objects are not thread-safe: one per camera thread


def _clahe():
    if not hasattr(_LOCAL, "clahe"):
        _LOCAL.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return _LOCAL.clahe
IRIS_MODEL_PATH = Path(__file__).resolve().parent / "models" / "iris_landmark.tflite"
IRIS_CROP_FRACTION = 0.55  # square eye crop side, fraction of frame width
IRIS_MIN_DARKNESS = 35.0   # gray levels darker than the search box median


class IrisModel:
    """MediaPipe iris-landmark model (64x64 eye crop -> 5 iris points), models/iris_landmark.tflite."""

    def __init__(self, model_path: Path = IRIS_MODEL_PATH) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]["index"]
        self._iris = self._interpreter.get_output_details()[1]["index"]

    @classmethod
    def load(cls) -> Optional["IrisModel"]:
        try:
            return cls() if IRIS_MODEL_PATH.exists() else None
        except Exception:  # no runtime: classic detector only
            return None

    def predict(self, image: np.ndarray, center: tuple[float, float], mirror: bool) -> tuple[float, float, float]:
        """Return iris center x, y and radius in pixels for a crop centered on `center`."""
        height, width = image.shape[:2]
        side = int(width * IRIS_CROP_FRACTION)
        cx, cy = int(center[0] * width), int(center[1] * height)
        padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
        left, top = cx - side // 2, cy - side // 2
        crop = padded[top + side: top + 2 * side, left + side: left + 2 * side]
        tensor = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (64, 64)).astype(np.float32) / 255.0
        if mirror:
            tensor = tensor[:, ::-1]
        self._interpreter.set_tensor(self._input, np.ascontiguousarray(tensor[None]))
        self._interpreter.invoke()
        iris = self._interpreter.get_tensor(self._iris).reshape(5, 3)[:, :2].copy()
        if mirror:
            iris[:, 0] = 64 - iris[:, 0]
        scale = side / 64
        radius = float(np.linalg.norm(iris[1] - iris[3])) * scale / 2
        return float(iris[0, 0] * scale + left), float(iris[0, 1] * scale + top), radius


def _darkness(gray: np.ndarray, x: float, y: float, reference: float) -> float:
    xi, yi = int(x), int(y)
    if not (0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]):
        return 0.0
    patch = gray[max(0, yi - 4): yi + 5, max(0, xi - 4): xi + 5]
    return reference - float(patch.mean())


@dataclass(frozen=True)
class PupilResult:
    locked: bool
    x: Optional[float]           # normalized frame coordinates
    y: Optional[float]
    confidence: float
    brightness: float
    contour: Optional[np.ndarray]  # full-frame pixel contour for drawing
    box: tuple[int, int, int, int]  # search box x0, y0, x1, y1 in pixels
    source: str = "classic"
    radius: float = 0.0


def search_box(width: int, height: int, aim: Optional[tuple[float, float]]) -> tuple[int, int, int, int]:
    if aim is None:
        return (int(width * FULL_FRAME_MARGIN), int(height * FULL_FRAME_MARGIN),
                int(width * (1 - FULL_FRAME_MARGIN)), int(height * (1 - FULL_FRAME_MARGIN)))
    half_w, half_h = width * AIM_BOX_WIDTH / 2, height * AIM_BOX_HEIGHT / 2
    cx = min(max(aim[0] * width, half_w), width - half_w)
    cy = min(max(aim[1] * height, half_h), height - half_h)
    return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)


def _touches_inner_border(contour: np.ndarray, roi_shape: tuple[int, int], open_sides: tuple[bool, ...]) -> bool:
    """True when the blob touches a search-box side that is not the image edge."""
    x, y, w, h = cv2.boundingRect(contour)
    hits = (x <= 1, y <= 1, x + w >= roi_shape[1] - 1, y + h >= roi_shape[0] - 1)
    return any(hit and not image_edge for hit, image_edge in zip(hits, open_sides))


def _candidate_score(contour, roi, blur, previous_px, roi_area, open_sides) -> Optional[tuple[float, float, float, np.ndarray]]:
    area = float(cv2.contourArea(contour))
    if not (roi_area * MIN_AREA_FRACTION <= area <= roi_area * MAX_AREA_FRACTION):
        return None
    if _touches_inner_border(contour, roi.shape, open_sides):
        return None
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = area / hull_area
    _, _, bw, bh = cv2.boundingRect(contour)
    aspect = bw / max(1.0, float(bh))
    if solidity < 0.55 or not 0.4 <= aspect <= 2.6:
        return None
    mask = np.zeros_like(roi)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    ring = cv2.subtract(cv2.dilate(mask, RING_KERNEL), mask)
    inside = float(cv2.mean(blur, mask=mask)[0])
    outside = float(cv2.mean(blur, mask=ring)[0])
    contrast = outside - inside
    if contrast < 12:
        return None
    ys, xs = np.nonzero(mask)
    weights = (255.0 - blur[ys, xs].astype(np.float64)) ** 3
    total = float(weights.sum()) or 1.0
    cx, cy = float((xs * weights).sum() / total), float((ys * weights).sum() / total)
    roundness = min(aspect, 1 / aspect)
    continuity = 0.5
    if previous_px is not None:
        distance = np.hypot(cx - previous_px[0], cy - previous_px[1]) / max(roi.shape)
        continuity = max(0.0, 1.0 - distance / PREVIOUS_RADIUS)
    size = min(1.0, area / (roi_area * 0.03))
    score = (min(1.0, contrast / 60.0) * 0.30 + solidity * 0.15 + roundness * 0.10
             + size * 0.20 + continuity * 0.25)
    return score, cx, cy, hull


def detect_pupil(image: np.ndarray, aim: Optional[tuple[float, float]],
                 previous: Optional[tuple[float, float]], iris_model: Optional[IrisModel] = None) -> PupilResult:
    """Classic detection, then the iris model on the crop and its mirror; the darker centre wins."""
    classic = detect_classic(image, aim, previous)
    if iris_model is None or aim is None:
        return classic
    center = previous or ((classic.x, classic.y) if classic.locked else aim)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = classic.box
    reference = float(np.median(gray[y0:y1, x0:x1]))
    height, width = gray.shape
    try:
        candidates = [iris_model.predict(image, center, mirror) for mirror in (False, True)]
    except Exception:
        return classic
    model_dark, mx, my, radius = max((_darkness(gray, x, y, reference), x, y, r) for x, y, r in candidates)
    classic_dark = _darkness(gray, classic.x * width, classic.y * height, reference) if classic.locked else 0.0
    inside = x0 <= mx < x1 and y0 <= my < y1
    if not inside or model_dark < IRIS_MIN_DARKNESS or model_dark < classic_dark:
        return classic
    confidence = round(min(1.0, 0.5 + model_dark / 200.0), 3)
    return PupilResult(True, round(mx / width, 4), round(my / height, 4), confidence,
                       classic.brightness, None, classic.box, "model", radius)


def detect_classic(image: np.ndarray, aim: Optional[tuple[float, float]],
                   previous: Optional[tuple[float, float]]) -> PupilResult:
    """Find the iris/pupil in a BGR frame. `aim`/`previous` are normalized points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    box = search_box(width, height, aim)
    x0, y0, x1, y1 = box
    roi = gray[y0:y1, x0:x1]
    brightness = round(float(np.mean(gray)) / 255.0, 3)
    blur = cv2.GaussianBlur(_clahe().apply(roi), (7, 7), 0)
    roi_area = float(roi.shape[0] * roi.shape[1])
    previous_px = None if previous is None else (previous[0] * width - x0, previous[1] * height - y0)
    open_sides = (x0 <= 0, y0 <= 0, x1 >= width, y1 >= height)

    best = None
    for percentile in DARK_PERCENTILES:
        mask = cv2.inRange(blur, 0, int(np.percentile(blur, percentile)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)   # drop lashes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)  # fill glints
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = _candidate_score(contour, roi, blur, previous_px, roi_area, open_sides)
            if candidate is not None and (best is None or candidate[0] > best[0]):
                best = candidate

    if best is None or best[0] < MIN_LOCK_SCORE:
        confidence = 0.0 if best is None else round(best[0], 3)
        return PupilResult(False, None, None, confidence, brightness, None, box)
    score, cx, cy, hull = best
    return PupilResult(True, round((cx + x0) / width, 4), round((cy + y0) / height, 4),
                       round(float(score), 3), brightness, hull + np.array([[[x0, y0]]]), box)


def annotate(image: np.ndarray, result: PupilResult, label: str, aimed: bool) -> bytes:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    white, grey, black = (255, 255, 255), (150, 150, 150), (0, 0, 0)
    x0, y0, x1, y1 = result.box
    cv2.rectangle(annotated, (x0, y0), (x1, y1), white if aimed else grey, 1)
    if result.locked:
        px, py = int(result.x * width), int(result.y * height)
        if result.contour is not None:
            cv2.drawContours(annotated, [result.contour], -1, white, 2)
        else:
            cv2.circle(annotated, (px, py), max(8, int(result.radius)), white, 2)
        cv2.circle(annotated, (px, py), 6, white, -1)
        cv2.line(annotated, (px - 30, py), (px + 30, py), white, 1)
        cv2.line(annotated, (px, py - 30), (px, py + 30), white, 1)
    source = " ML" if result.source == "model" else ""
    state = f"LOCK{source} {result.confidence:.2f}" if result.locked else "NO LOCK"
    hint = "" if aimed else " // CALIBRATE TO AIM"
    cv2.rectangle(annotated, (0, 0), (width, 40), black, -1)
    cv2.putText(annotated, f"{label} // {state}{hint}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, white, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if ok else b""
