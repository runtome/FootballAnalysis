from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def create_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps or 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer


def frame_limit_reached(frame_index: int, max_frames: Optional[int]) -> bool:
    return max_frames is not None and frame_index >= max_frames
