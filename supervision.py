import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


def main():
  #Path to video
  video_path = "input/55c9d1_4.mp4"

  #Read video
  cap = cv2.VideoCapture(video_path)

  model = YOLO("yolov8l.pt")
  tracker = sv.ByteTrack()
  
  box_annotator = sv.BoxAnnotator(
      thickness=2,
      text_thickness=2,
      text_scale=1
  )

  while True:
      ret, frame = cap.read()

      result = model(frame, agnostic_nms=True)[0]
      detections = sv.Detections.from_yolov8(result)
      labels = [
          f"{model.model.names[class_id]} {confidence:0.2f}"
          for _, confidence, class_id, _
          in detections
      ]
      frame = box_annotator.annotate(
          scene=frame, 
          detections=detections, 
          labels=labels
      )

      cv2.imshow("yolov8", frame)

      if (cv2.waitKey(30) == ord('q')):
          break


if __name__ == "__main__":
    main()