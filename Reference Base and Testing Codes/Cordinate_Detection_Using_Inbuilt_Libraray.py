import math

import cv2
import numpy as np
import pyrealsense2 as rs  # Import the RealSense library
from ultralytics import YOLO

model_path = r"G:\Grid5.0\Detection Models\BoxDetection.pt"
class_names = ["box"]
confidence_threshold = 0.8
model = YOLO(model_path)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    results = model(color_image)

    detected_objects = []
    for r in results:
        for box in r.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf >= 0.85:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

                if 0 <= centroid_x < depth_image.shape[1] and 0 <= centroid_y < depth_image.shape[0]:
                    depth = depth_frame.get_distance(centroid_x, centroid_y)

                    x3d = (centroid_x - depth_intrin.ppx) / depth_intrin.fx * depth
                    y3d = (centroid_y - depth_intrin.ppy) / depth_intrin.fy * depth
                    z_3d = depth

                    class_idx = int(box.cls[0])
                    class_name = class_names[class_idx]
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    cv2.putText(color_image, f"Distance: {z_3d:.2f} m", (centroid_x, centroid_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(color_image, f"3D Coordinates: ({x3d:.2f}, {y3d:.2f}, {z_3d:.2f})",
                                (centroid_x, centroid_y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Object Detection, Centroid, and 3D Coordinates', color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
