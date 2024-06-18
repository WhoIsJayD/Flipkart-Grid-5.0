import cv2
from ultralytics import YOLO


def detect_qrcode(video_sources, model_path, confidence_threshold=0.8, detect_every=5):
    model = YOLO(model_path)

    # Open video capture for each source
    caps = [cv2.VideoCapture(source) for source in video_sources]
    detect_counters = [0] * len(video_sources)

    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()

            # Resize frame to speed up detection
            frame_rgb = cv2.resize(frame, (320, 240))

            detect_counters[idx] += 1
            if detect_counters[idx] == detect_every:
                detect_counters[idx] = 0

                results = model(frame_rgb)

                for r in results:
                    for box in r.boxes:
                        conf = round(box.conf[0].item(), 2)
                        if conf >= confidence_threshold:
                            print(f"Detected in video source {idx + 1}")
                            cv2.putText(frame, f"Detected in source {idx + 1}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            break

                # Display the frame
                cv2.imshow(f"Source {idx + 1}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video captures and close all windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


video_sources = ["http://192.168.137.132:81/", "http://192.168.137.159:81/"]
model_path = r"G:\Grid5.0\Detection Models\QRCode_1.pt"
result = detect_qrcode(video_sources, model_path)
