from __future__ import annotations

from typing import Any

from src.detection.yolo_person_detector import PERSON_CLASS_ID


def track_frame(
    model: Any,
    frame,
    *,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
):
    return model.track(
        frame,
        persist=True,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]
