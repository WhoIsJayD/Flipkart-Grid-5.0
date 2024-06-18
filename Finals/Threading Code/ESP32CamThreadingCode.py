import time
from queue import Queue
from threading import Thread, Event

import cv2
from ultralytics import YOLO


class ObjectDetectionThread(Thread):
    def __init__(self, thread_index, model_path, video_source, stop_event, result_queue, confidence_threshold=0.7,
                 detect_every=3):
        super(ObjectDetectionThread, self).__init__()
        self.thread_index = thread_index
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.detect_every = detect_every
        self.stop_event = stop_event
        self.result_queue = result_queue

    def run(self):
        cap = cv2.VideoCapture(self.video_source)

        while not self.stop_event.is_set():
            ret, frame = cap.read()

            # Resize frame to speed up detection
            frame_rgb = cv2.resize(frame, (320, 240))

            results = self.model(frame_rgb)

            for r in results:
                for box in r.boxes:
                    conf = round(box.conf[0].item(), 2)
                    if conf >= self.confidence_threshold:
                        detection_data = {
                            'thread_index': self.thread_index,
                            'video_source': self.video_source,
                            'confidence': conf
                        }
                        self.result_queue.put(detection_data)
                        self.stop_event.set()

            time.sleep(0.1)

        cap.release()


def perform_object_detection(model_path, video_sources, confidence_threshold=0.7, detect_every=3):
    stop_event = Event()
    result_queue = Queue()
    threads = []

    for i, video_source in enumerate(video_sources, start=1):
        thread = ObjectDetectionThread(i, model_path, video_source, stop_event, result_queue, confidence_threshold,
                                       detect_every)
        threads.append(thread)
        thread.start()

    # Optionally, you may want to wait for the threads to finish before exiting
    for thread in threads:
        thread.join()

    print("Stopping all threads as QR code detected.")

    # Retrieve detection results from the queue
    detection_results = []
    while not result_queue.empty():
        detection_data = result_queue.get()
        detection_results.append(detection_data)

    return detection_results


if __name__ == "__main__":
    model_path = r"G:\Grid5.0\Detection Models\QRCode.pt"
    video_sources = ["http://192.168.137.193:81/", "http://192.168.137.163:81/"]

    results = perform_object_detection(model_path, video_sources)
    print(results)
    for result in results:
        print(f"Detected QR code in Thread {result['thread_index']} "
              f"from source {result['video_source']} with confidence {result['confidence']}")
