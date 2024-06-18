import cv2
from ultralytics import YOLO

# Use a smaller YOLOv5s model
model = YOLO(r"G:\Grid5.0\Detection Models\QRCode.pt")

confidence_threshold = 0.8
detect_every = 5
detect_counter = 0

cap = cv2.VideoCapture("http://192.168.137.132:81/")

while True:
    ret, frame = cap.read()

    # Resize frame to speed up detection
    frame_rgb = cv2.resize(frame, (320, 240))

    detect_counter += 1
    if detect_counter == detect_every:
        detect_counter = 0

        results = model(frame_rgb)

        for r in results:
            for box in r.boxes:
                conf = round(box.conf[0].item(), 2)
                if conf >= confidence_threshold:
                    print("Detected")
                    exit(0)