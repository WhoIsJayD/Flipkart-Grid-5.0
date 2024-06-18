import logging
import math
import threading
import time

import cv2
import pyrealsense2 as rs
import serial
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ObjectDetector:
    def __init__(self, model_path, class_names, confidence_threshold=0.85, serial_port='COM6', baud_rate=115200):
        try:
            self.model = YOLO(model_path)
            self.class_names = class_names
            self.confidence_threshold = confidence_threshold

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

            self.ser = serial.Serial(serial_port, baud_rate)
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise e

    def write_to_arduino(self, data):
        try:
            self.ser.write(data.encode() + b'\n')
        except Exception as e:
            logging.error(f"Error writing to Arduino: {e}")

    def read_from_arduino(self):
        try:
            data = self.ser.readline().decode("utf-8").strip()
            while data != "DONE":
                logging.info(data)
                with open("file.arduino", "a") as f:
                    f.write(data + '\n')
                data = self.ser.readline().decode("utf-8").strip()
            return data
        except Exception as e:
            logging.error(f"Error reading from Arduino: {e}")
            return "DONE"

    def start_camera(self):
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            logging.error(f"Error starting camera pipeline: {e}")

    def stop_camera(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            logging.error(f"Error stopping camera pipeline: {e}")

    def get_depth_data(self):
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            return depth_frame
        except Exception as e:
            logging.error(f"Error getting depth data: {e}")

    def convert_pixel_to_3d(self, x, y, depth_frame):
        try:
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            z = depth_frame.get_distance(x, y)
            logging.info(f"Simple Coordinates: {x}, {y}, {z}")
            x3d = (x - depth_intrin.ppx) / depth_intrin.fx * z
            y3d = (y - depth_intrin.ppy) / depth_intrin.fy * z
            logging.info(f"Cartesian Coordinates: {x3d}, {y3d}, {z}")
            return x3d, y3d, z
        except Exception as e:
            logging.error(f"Error converting pixel to 3D: {e}")

    def detect_objects(self, frame):
        try:
            results = self.model(frame, stream=True)
            detected_objects = []

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

            return detected_objects
        except Exception as e:
            logging.error(f"Error detecting objects: {e}")

    def inverse_kinematics(self, x, y, z):
        try:
            # Inverse kinematics code
            d1 = 15
            d2 = 45
            d3 = 50
            d4 = 21
            d5 = 6

            nx = 1
            ny = 0
            nz = 0

            sx = 0
            sy = -1
            sz = 0

            ax = 0
            ay = 0
            az = -1

            q1 = math.atan2(y, x)

            eq1 = -(ax * math.cos(q1) + ay * math.sin(q1))
            eq2 = -az

            q5 = math.atan2((nx * math.sin(q1) - ny * math.cos(q1)), (sx * math.sin(q1) - sy * math.cos(q1)))

            c = (x / math.cos(q1)) + (d5 * eq1) - (d4 * eq2)
            d = (d1 - (d4 * eq1) - (d5 * eq2) - z)

            R = (c * c + d * d - d3 * d3 - d2 * d2) / (2 * (d3 * d2))
            t = math.sqrt(1 - R * R)

            q3 = math.atan2(t, R)

            r = d3 * math.cos(q3) + d2
            s = d3 * math.sin(q3)

            q2 = math.atan2((r * d) - (s * c), (r * c) + (s * d))

            eq3 = math.atan2(-(ax * math.cos(q1) + ay * math.sin(q1)), -az)
            q4 = eq3 - (q2 + q3)

            # Convert angles to degrees
            angle1 = math.degrees(q1)
            angle2 = math.degrees(q2)
            angle3 = math.degrees(q3)
            angle4 = math.degrees(q4)
            angle5 = math.degrees(q5)

            # Apply angle constraints
            angle1 = self.angle_constraint(angle1)
            angle2 = self.angle_constraint(angle2)
            angle3 = self.angle_constraint(angle3)
            angle4 = self.angle_constraint(angle4)
            angle5 = self.angle_constraint(angle5)

            # Apply demap for the 2nd motor
            angle2 = self.linear_convert(angle2, 0, 360, 0, 8192)

            return angle1, angle2, angle3, angle4, angle5
        except Exception as e:
            logging.error(f"Error in inverse kinematics: {e}")

    @staticmethod
    def angle_constraint(angle):
        if 0 > angle > -90:
            angle = angle + 90
        if 0 > angle < -90:
            angle = abs(angle) + 90
        if 90 < angle < 180:
            angle = angle - 90
        if 90 < angle > 180:
            angle = angle - 180  # or 270 - angle1
        return angle

    @staticmethod
    def linear_convert(value, input_min, input_max, output_min, output_max):
        demap_angle = (value - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
        return demap_angle

    def send_data_to_arduino(self, x, y, z):
        try:
            inverse_angles = self.inverse_kinematics(x, y, z)
            motor_positions = f'MOTOR,{inverse_angles[0]:.2f},{inverse_angles[1]:.2f},{inverse_angles[2]:.2f},{inverse_angles[3]:.2f},{inverse_angles[4]:.2f}'
            self.write_to_arduino(motor_positions)
        except Exception as e:
            logging.error(f"Error sending data to Arduino: {e}")

    def process_frames(self, cap, arduino_delay=0.01):
        try:
            processing_paused = False
            last_frame_with_closest_object = None

            def detection_thread():
                nonlocal processing_paused, last_frame_with_closest_object
                while True:
                    if not processing_paused:
                        success, img = cap.read()
                        img = cv2.resize(img, (1024, 768))
                        if not success:
                            break

                        detected_objects = self.detect_objects(img)

                        closest_object = None
                        min_distance = float('inf')

                        for obj in detected_objects:
                            x, y, z = obj["x3d"], obj["y3d"], obj["z3d"]

                            if 0 < z < min_distance:
                                min_distance = z
                                closest_object = obj

                        if len(detected_objects) > 1 and closest_object is not None and closest_object["z3d"] > 0:
                            processing_paused = True
                            last_frame_with_closest_object = img.copy()

                    time.sleep(arduino_delay)

            detection_thread = threading.Thread(target=detection_thread, daemon=True)
            detection_thread.start()

            while True:
                if processing_paused and last_frame_with_closest_object is not None:
                    x, y, z = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
                    self.send_data_to_arduino(x, y, z)

                    cv2.imshow("Object Detection", last_frame_with_closest_object)
                    if self.read_from_arduino() == "DONE":
                        with open("file.arduino", "a") as f:
                            f.write("lol\n")
                        processing_paused = False
                else:
                    success, img = cap.read()
                    img = cv2.resize(img, (1024, 768))
                    if not success:
                        break

                    cv2.imshow("Object Detection", img)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    processing_paused = not processing_paused
        except Exception as e:
            logging.error(f"Error processing frames: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop_camera()

    def run_detection(self, cap_device=1):
        try:
            model_path = "model1.pt"
            class_names = ["box"]
            confidence_threshold = 0.90
            cap = cv2.VideoCapture(cap_device)

            self.start_camera()

            self.process_frames(cap)
        except Exception as e:
            logging.error(f"Error running detection: {e}")
        finally:
            self.stop_camera()


if __name__ == "__main__":
    try:
        detector = ObjectDetector("model1.pt", ["box"], 0.90, serial_port='COM6', baud_rate=115200)
        detector.run_detection(cap_device=1)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
