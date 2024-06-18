import math
import time

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO(r"model1.pt")

classNames = ["box"]

prev_frame_time = time.time()
new_frame_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    new_frame_time = time.time()

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf * 100 >= 85:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Calculate the center coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Display the detected object coordinates and center
                text = f'Coordinates: ({center_x}, {center_y})'
                cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  # Draw a red circle at the center

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
