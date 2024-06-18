import math

import cv2
from ultralytics import YOLO

detection = True  # Set to True initially
model = YOLO(r'G:\Grid5.0\Detection Models\QRCode_2.pt')
class_names = ["QR_CODE"]
confidence_threshold = 0.60
cap = cv2.VideoCapture("http://192.168.137.159:81/")
cap.set(3, 1280)
cap.set(4, 720)
detected = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf >= 0.75:
                    print("Detected")
                    detection = False

    cv2.imshow('ESP 2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
