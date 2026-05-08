from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class PitchPoint:
    x_norm: float
    y_norm: float
    in_field: bool


@dataclass(frozen=True)
class FieldBoundaries:
    xs: np.ndarray
    upper: np.ndarray
    lower: np.ndarray

    def y_at(self, x: float) -> Tuple[float, float]:
        upper_y = float(np.interp(x, self.xs, self.upper))
        lower_y = float(np.interp(x, self.xs, self.lower))
        return upper_y, lower_y


def load_polygon(raw_polygon: Optional[Iterable[Sequence[float]]]) -> Optional[np.ndarray]:
    if not raw_polygon:
        return None
    points = [(float(point[0]), float(point[1])) for point in raw_polygon]
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float32)


def foot_point(xyxy: Sequence[float]) -> Point:
    x1, _, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, y2)


def estimate_field_boundaries(frame: np.ndarray, bins: int = 32) -> Optional[FieldBoundaries]:
    """Estimate the visible pitch upper/lower borders from green pixels."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (95, 255, 220))
    kernel = np.ones((7, 7), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)

    xs = np.linspace(0, width - 1, bins, dtype=np.float32)
    upper = np.full(bins, np.nan, dtype=np.float32)
    lower = np.full(bins, np.nan, dtype=np.float32)
    half_bin = max(8, width // bins)

    for idx, x in enumerate(xs):
        left = max(0, int(x) - half_bin)
        right = min(width, int(x) + half_bin)
        column = green[:, left:right]
        ys = np.where(column > 0)[0]
        if len(ys) < height * 0.08:
            continue
        upper[idx] = float(np.percentile(ys, 3))
        lower[idx] = float(np.percentile(ys, 97))

    valid = np.isfinite(upper) & np.isfinite(lower)
    if int(valid.sum()) < max(6, bins // 5):
        return None

    upper = np.interp(xs, xs[valid], upper[valid]).astype(np.float32)
    lower = np.interp(xs, xs[valid], lower[valid]).astype(np.float32)

    smooth_kernel = np.ones(5, dtype=np.float32) / 5.0
    upper = np.convolve(upper, smooth_kernel, mode="same")
    lower = np.convolve(lower, smooth_kernel, mode="same")
    upper[:2] = upper[2]
    upper[-2:] = upper[-3]
    lower[:2] = lower[2]
    lower[-2:] = lower[-3]

    min_gap = max(80.0, height * 0.18)
    lower = np.maximum(lower, upper + min_gap)
    lower = np.minimum(lower, height - 1.0)
    return FieldBoundaries(xs=xs, upper=upper, lower=lower)


def map_to_pitch(
    xyxy: Sequence[float],
    frame_width: int,
    frame_height: int,
    field_polygon: Optional[np.ndarray] = None,
    field_boundaries: Optional[FieldBoundaries] = None,
    boundary_margin_px: float = 0.0,
) -> PitchPoint:
    """Rule-based frame-to-mini-pitch mapping for pan/zoom broadcast footage.

    This is not real homography. It uses the player's foot point and normalizes
    it against the visible frame or optional field polygon bounding rectangle.
    """
    px, py = foot_point(xyxy)
    in_field = True

    if field_boundaries is not None:
        top, bottom = field_boundaries.y_at(px)
        margin = max(0.0, float(boundary_margin_px))
        in_field = (top - margin) <= py <= (bottom + margin)
        x_norm = px / max(float(frame_width), 1.0)
        y_norm = (py - top) / max(bottom - top, 1.0)
        return PitchPoint(
            x_norm=float(np.clip(x_norm, 0.0, 1.0)),
            y_norm=float(np.clip(y_norm, 0.0, 1.0)),
            in_field=in_field,
        )

    if field_polygon is not None:
        in_field = cv2.pointPolygonTest(field_polygon, (px, py), False) >= 0
        x, y, w, h = cv2.boundingRect(field_polygon.astype(np.int32))
        left, top = float(x), float(y)
        right, bottom = float(x + max(w, 1)), float(y + max(h, 1))
    else:
        left, top = 0.0, 0.0
        right, bottom = float(max(frame_width, 1)), float(max(frame_height, 1))

    x_norm = (px - left) / max(right - left, 1.0)
    y_norm = (py - top) / max(bottom - top, 1.0)
    return PitchPoint(
        x_norm=float(np.clip(x_norm, 0.0, 1.0)),
        y_norm=float(np.clip(y_norm, 0.0, 1.0)),
        in_field=in_field,
    )
