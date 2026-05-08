from __future__ import annotations

from ultralytics import YOLO


PERSON_CLASS_ID = 0


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)
