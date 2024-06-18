import time
from threading import Thread

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
        self.flag = 0
    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        a = time.time()
        while not self.stop_event.is_set():
            ret, frame = cap.read()

            # Resize frame to speed up detection
            frame_rgb = cv2.resize(frame, (320, 240))

            results = self.model(frame_rgb)

            for r in results:
                for box in r.boxes:
                    conf = round(box.conf[0].item(), 2)
                    if conf >= self.confidence_threshold:
                        self.flag = True

                        detection_data = {

                            'thread_index': self.thread_index,
                            'video_source': self.video_source,
                            'confidence': conf
                        }
                        self.result_queue.put(detection_data)
                        self.stop_event.set()
            if (time.time() - a >= 10):
                return self.flag
            time.sleep(0.1)

        cap.release()


class ObjectDetectionThread2(Thread):
    def __init__(self, thread_index, model_path, video_source, stop_event, result_queue, confidence_threshold=0.7,
                 detect_every=3):
        super(ObjectDetectionThread2, self).__init__()
        self.thread_index = thread_index
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.detect_every = detect_every
        self.stop_event = stop_event
        self.result_queue = result_queue
        self.flag = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        a = time.time()
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

                            'thread_index': 2,
                            'video_source': self.video_source,
                            'confidence': conf
                        }
                        self.result_queue.put(detection_data)
                        self.stop_event.set()
            if (time.time() - a >= 10):
                return self.flag
            time.sleep(0.1)

        cap.release()
