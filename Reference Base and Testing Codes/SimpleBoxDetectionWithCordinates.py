import math

import cv2
from ultralytics import YOLO

model_path = "model2.pt"
class_names = ["flyer", "mediumBox", "smallBox"]
confidence_threshold = 0.70
model = YOLO(model_path)

cap = cv2.VideoCapture("http://192.168.137.252:81")
cap.set(3, 1280)
cap.set(4, 720)
while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf >= 0.85:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                class_idx = int(box.cls[0])
                class_name = class_names[class_idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)

    cv2.imshow('Object Detection and Centroid', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
