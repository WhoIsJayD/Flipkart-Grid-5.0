import math
from collections import OrderedDict

import cv2
import numpy as np
from ultralytics import YOLO


# Define the CentroidTracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            for i in range(0, len(objectCentroids)):
                for j in range(0, len(inputCentroids)):
                    D[i, j] = np.sqrt(np.sum((objectCentroids[i] - inputCentroids[j]) ** 2))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


# Define the ObjectDetector class
class ObjectDetector:

    def __init__(self, model_path, class_names, confidence_threshold=0.85):
        self.fov = 80
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.tracker = CentroidTracker()

    def detect_objects(self, frame):
        results = self.model(frame, stream=True)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    class_idx = int(box.cls[0])
                    class_name = self.class_names[class_idx]
                    detected_objects.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "center_x": center_x,
                        "center_y": center_y,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

        return detected_objects
    # def distance(self, frame) :
    #     return self.distance
    # def cordinate(self, frame) :
    #     dp = self.distance * math.tan(self.fov/2) / self.resolution
    #     return dp


def main():
    model_path = "model1.pt"
    class_names = ["box"]
    confidence_threshold = 0.85

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = ObjectDetector(model_path, class_names, confidence_threshold)

    output_file = open("object_data.txt", "w")

    while True:
        success, img = cap.read()
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if not success:
            break

        detected_objects = detector.detect_objects(img)

        rects = []
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            rects.append((x1, y1, x2, y2))

        tracked_objects = detector.tracker.update(rects)

        for obj_id, centroid in tracked_objects.items():
            x, y = centroid
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            output_str = f'Object ID: {obj_id}, Center: ({x:.2f},{y:.2f})\n'
            output_file.write(output_str)
            output_file.flush()
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Object Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    output_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
