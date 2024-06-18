# object_detector.py
import math

import pyrealsense2 as rs
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, class_names, confidence_threshold=0.85):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)

    def start_camera(self):
        self.pipeline.start(self.config)

    def stop_camera(self):
        self.pipeline.stop()

    def get_depth_data(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def convert_pixel_to_3d(self, x, y, depth_frame):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        z = depth_frame.get_distance(x, y)
        print("Simple Coordinates : ", x, y, z)
        x3d = (x - depth_intrin.ppx) / depth_intrin.fx * z
        y3d = (y - depth_intrin.ppy) / depth_intrin.fy * z
        print("Cartesian Coordinates : ", x3d, y3d, z)
        return x3d, y3d, z

    def detect_objects(self, frame):
        results = self.model(frame, stream=True)
        detected_objects = []

        # Get depth data from the Lidar camera
        depth_frame = self.get_depth_data()

        for r in results:
            for box in r.boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    x3d, y3d, z3d = self.convert_pixel_to_3d(centroid_x, centroid_y, depth_frame)

                    class_idx = int(box.cls[0])
                    class_name = self.class_names[class_idx]
                    detected_objects.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "center_x": centroid_x,
                        "center_y": centroid_y,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x3d": x3d,
                        "y3d": y3d,
                        "z3d": z3d
                    })
                    print(detected_objects)

        return detected_objects
