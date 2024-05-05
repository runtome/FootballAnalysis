from ultralytics import YOLO
import numpy as np
import cv2

from utils import get_center_of_bbox, get_bbox_width

import sys
sys.path.append('../')


class Tracker:
  def __init__(self, model_path) -> None:
      self.model = YOLO(model_path)

  def draw_player(self, frame ):

    frame = frame.copy()
     
    # Predict frame 
    result = self.model(frame)[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')
    confidences = np.array(result.boxes.conf.cpu())
    class_name = result.names
    conf_threshold = 0
    color =  (255, 0, 0)

    for cls, bbox, conf in zip(classes, bboxes, confidences):
      if conf > conf_threshold:
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x_center, y_center = get_center_of_bbox(bbox)
          width = get_bbox_width(bbox)

          # Draw ellipse
          cv2.ellipse(
              frame,
              center=(x_center, y2),
              axes=(width, int(0.35 * width)),
              angle=0.0,
              startAngle=-45,
              endAngle=235,
              color=color,
              thickness=2,
              lineType=cv2.LINE_4
          )

    return frame

     
     