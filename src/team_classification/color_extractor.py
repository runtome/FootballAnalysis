from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


ColorFeature = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ShirtColor:
    hsv: Tuple[float, float, float]
    feature: ColorFeature


def extract_shirt_color(frame: np.ndarray, xyxy: Sequence[float]) -> Optional[ShirtColor]:
    """Extract a compact HSV-based color feature from the player's upper body."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 12 or box_h < 24:
        return None

    # Upper torso area: avoid shorts, socks, and as much grass as possible.
    torso_y1 = y1 + int(box_h * 0.18)
    torso_y2 = y1 + int(box_h * 0.58)
    torso_x1 = x1 + int(box_w * 0.18)
    torso_x2 = x2 - int(box_w * 0.18)
    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    if pixels.size == 0:
        return None

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]

    # Prefer colored kit pixels and suppress field-green background leakage.
    colored = (sat > 35) & (val > 35)
    green_field = (hue >= 35) & (hue <= 88) & (sat > 45)
    mask = colored & ~green_field
    if int(mask.sum()) < max(20, pixels.shape[0] // 20):
        mask = colored
    if int(mask.sum()) < 10:
        return None

    selected = pixels[mask]
    median_hsv = np.median(selected, axis=0)
    h_norm = float(median_hsv[0] / 179.0)
    s_norm = float(median_hsv[1] / 255.0)
    v_norm = float(median_hsv[2] / 255.0)

    # Circular hue representation lets red values near 0/179 cluster correctly.
    hue_angle = 2.0 * np.pi * h_norm
    feature = (
        float(np.sin(hue_angle) * max(s_norm, 0.2)),
        float(np.cos(hue_angle) * max(s_norm, 0.2)),
        s_norm,
        v_norm,
    )
    return ShirtColor(
        hsv=(float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])),
        feature=feature,
    )
