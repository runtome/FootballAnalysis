from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from src.field.pitch_mapper import FieldBoundaries


TEAM_COLORS = {
    "A": (40, 190, 255),
    "B": (255, 90, 40),
    "U": (180, 180, 180),
}


@dataclass(frozen=True)
class MiniPitchPlayer:
    x_norm: float
    y_norm: float
    team: str
    label: str
    in_field: bool = True


def draw_player(
    frame: np.ndarray,
    xyxy: Sequence[float],
    label: str,
    team: str,
    confidence: float,
    show_confidence: bool = False,
    speed_kmh: float = 0.0,
    distance_m: float = 0.0,
    draw_bbox: bool = False,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    color = TEAM_COLORS.get(team, TEAM_COLORS["U"])
    if draw_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    number_text = _compact_team_label(label)
    if show_confidence:
        number_text = f"{number_text} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    center_x = (x1 + x2) // 2

    font_scale = 0.52
    thickness = 2
    (num_w, num_h), baseline = cv2.getTextSize(number_text, font, font_scale, thickness)
    foot_y = min(frame.shape[0] - 1, y2)
    ellipse_axes = (max(8, int((x2 - x1) * 0.4)), max(5, int((x2 - x1) * 0.16)))
    cv2.ellipse(frame, (center_x, foot_y), ellipse_axes, 0, -45, 225, color, 2, cv2.LINE_AA)

    label_y = max(num_h + 2, y1 - 6)
    _draw_centered_text(frame, number_text, center_x, label_y, font_scale, thickness, color)

    speed_text = f"{speed_kmh:.2f} km/h"
    distance_text = f"{distance_m:.2f} m"
    stat_scale = 0.43
    stat_thickness = 2
    stat_y = min(frame.shape[0] - 8, foot_y + ellipse_axes[1] + 18)
    _draw_centered_text(frame, speed_text, center_x, stat_y, stat_scale, stat_thickness)
    _draw_centered_text(frame, distance_text, center_x, min(frame.shape[0] - 8, stat_y + 17), stat_scale, stat_thickness)


def _compact_team_label(label: str) -> str:
    prefix = label[0] if label else ""
    digits = "".join(ch for ch in label if ch.isdigit())
    if not digits:
        return label
    if prefix in {"A", "B"}:
        return f"{prefix}{int(digits)}"
    return str(int(digits))


def _draw_centered_text(
    frame: np.ndarray,
    text: str,
    center_x: int,
    baseline_y: int,
    font_scale: float,
    thickness: int,
    color: Tuple[int, int, int] = (10, 10, 10),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(center_x - text_w / 2)
    cv2.putText(frame, text, (x + 1, baseline_y + 1), font, font_scale, (245, 245, 245), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, baseline_y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, frame_index: int, fps: float, team_counts: Tuple[int, int, int]) -> None:
    seconds = frame_index / fps if fps > 0 else 0.0
    a_count, b_count, u_count = team_counts
    text = f"POC v1 | {seconds:7.1f}s | Team A: {a_count}  Team B: {b_count}  Unknown: {u_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (12, 12), (760, 48), (30, 30, 30), -1)
    cv2.putText(frame, text, (24, 37), font, 0.65, (245, 245, 245), 2, cv2.LINE_AA)


def draw_field_boundaries(frame: np.ndarray, boundaries: FieldBoundaries) -> None:
    upper_points = np.column_stack((boundaries.xs, boundaries.upper)).astype(np.int32)
    lower_points = np.column_stack((boundaries.xs, boundaries.lower)).astype(np.int32)
    cv2.polylines(frame, [upper_points], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [lower_points], False, (255, 255, 255), 2, cv2.LINE_AA)


def draw_runtime_fps(frame: np.ndarray, runtime_fps: float) -> None:
    h, _ = frame.shape[:2]
    text = f"FPS: {runtime_fps:5.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = 14
    y = h - 16
    cv2.rectangle(
        frame,
        (x - 6, y - text_h - baseline - 8),
        (x + text_w + 8, y + baseline + 4),
        (25, 25, 25),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)


def draw_mini_pitch(
    frame: np.ndarray,
    players: Iterable[MiniPitchPlayer],
    width: int = 300,
    height: int = 185,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    margin = 16
    pitch_w = min(width, frame_w - margin * 2)
    pitch_h = min(height, frame_h - margin * 2)
    x0 = frame_w - pitch_w - margin
    y0 = frame_h - pitch_h - margin
    x1 = frame_w - margin
    y1 = frame_h - margin

    cv2.rectangle(frame, (x0, y0), (x1, y1), (34, 112, 54), -1)

    line_color = (235, 245, 235)
    cv2.rectangle(frame, (x0, y0), (x1, y1), line_color, 2)
    mid_x = x0 + pitch_w // 2
    cv2.line(frame, (mid_x, y0), (mid_x, y1), line_color, 1)
    cv2.circle(frame, (mid_x, y0 + pitch_h // 2), max(10, pitch_h // 7), line_color, 1)

    box_w = max(32, int(pitch_w * 0.18))
    box_h = max(70, int(pitch_h * 0.55))
    box_y0 = y0 + (pitch_h - box_h) // 2
    cv2.rectangle(frame, (x0, box_y0), (x0 + box_w, box_y0 + box_h), line_color, 1)
    cv2.rectangle(frame, (x1 - box_w, box_y0), (x1, box_y0 + box_h), line_color, 1)

    goal_h = max(34, int(pitch_h * 0.25))
    goal_y0 = y0 + (pitch_h - goal_h) // 2
    cv2.rectangle(frame, (x0, goal_y0), (x0 + max(12, box_w // 3), goal_y0 + goal_h), line_color, 1)
    cv2.rectangle(frame, (x1 - max(12, box_w // 3), goal_y0), (x1, goal_y0 + goal_h), line_color, 1)

    for player in players:
        px = x0 + int(round(player.x_norm * pitch_w))
        py = y0 + int(round(player.y_norm * pitch_h))
        color = TEAM_COLORS.get(player.team, TEAM_COLORS["U"])
        radius = 5 if player.in_field else 4
        thickness = -1 if player.in_field else 1
        cv2.circle(frame, (px, py), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)
