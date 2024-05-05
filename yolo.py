import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

image = cv2.imread('input\screenshot.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model = YOLO('models/best.pt')
results = model(image)

# print(results[0].boxes)

result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
classes = np.array(result.boxes.cls.cpu(), dtype='int')
confidences = np.array(result.boxes.conf.cpu())
class_name = result.names
conf_threshold = 0

# Dictionary to store label counts
label_counts = {}

for cls, bbox, conf in zip(classes, bboxes, confidences):
    if conf > conf_threshold:
        x1, y1, x2, y2 = bbox
        label = class_name[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Update label counts
        label_counts[label] = label_counts.get(label, 0) + 1

# White image information
cv2.putText(image, "data", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# Write label counts on the image
text_y = 40
for label, count in label_counts.items():
    text = f"{label}: {count}"
    cv2.putText(image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    text_y += 20

cv2.imshow('frame', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
