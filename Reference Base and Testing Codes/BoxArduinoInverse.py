import math

import cv2
import pyrealsense2 as rs
import serial
from ultralytics import YOLO

ser = serial.Serial('COM6', 115200)


def write_to_arduino(data):
    ser.write(data.encode() + b'\n')


def read_from_arduino():
    data = ser.readline().decode("utf-8").strip()
    while data != "DONE":
        print(data)
        data = ser.readline().decode("utf-8").strip()
    return data


# Defining Link Lengths
d1 = 15
d2 = 45
d3 = 50
d4 = 21
d5 = 6

# Defining Rotation Matrix
nx = 1
ny = 0
nz = 0

sx = 0
sy = -1
sz = 0

ax = 0
ay = 0
az = -1


# Defining Angle Constraints Function
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


# Defining Demap Function for 2nd Motor
def linear_convert(value, input_min, input_max, output_min, output_max):
    demap_angle = (value - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
    return demap_angle


# Inverse Kinematics Function
def inverse_kinematics(x, y, z):
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
    angle1 = angle_constraint(angle1)
    angle2 = angle_constraint(angle2)
    angle3 = angle_constraint(angle3)
    angle4 = angle_constraint(angle4)
    angle5 = angle_constraint(angle5)

    # Apply demap for the 2nd motor
    angle2 = linear_convert(angle2, 0, 360, 0, 8192)

    return angle1, angle2, angle3, angle4, angle5


class ObjectDetector:
    def __init__(self, model_path, class_names, confidence_threshold=0.85):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

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

        return detected_objects


def main():
    model_path = "model1.pt"
    class_names = ["box"]
    confidence_threshold = 0.90
    cap = cv2.VideoCapture(1)

    detector = ObjectDetector(model_path, class_names, confidence_threshold)
    detector.start_camera()

    processing_paused = False  # Variable to control frame processing
    last_frame_with_closest_object = None

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1024, 768))
        if not success:
            break

        if not processing_paused:
            detected_objects = detector.detect_objects(img)

            # Filter objects by distance and keep the closest one
            closest_object = None
            min_distance = float('inf')

            for obj in detected_objects:
                x, y, z = obj["x3d"], obj["y3d"], obj["z3d"]

                if 0 < z < min_distance:
                    min_distance = z
                    closest_object = obj

            # Check if there is more than one detected object and none of them have a distance of zero
            if len(detected_objects) > 1 and closest_object is not None and closest_object["z3d"] > 0:
                processing_paused = True
                last_frame_with_closest_object = img.copy()

                # Calculate inverse kinematics for the closest object
                x_ik, y_ik, z_ik = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
                inverse_angles = inverse_kinematics(x_ik, y_ik, z_ik)
                theta1, theta2, theta3, theta4, theta5 = inverse_angles

                # Send the calculated angles to Arduino
                motor_positions = f'MOTOR,{theta1:.2f},{theta2:.2f},{theta3:.2f},{theta4:.2f},{theta5:.2f}'
                write_to_arduino(motor_positions)

        # Display information of the closest object
        if closest_object:
            x, y, z = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
            cv2.rectangle(img, (closest_object["x1"], closest_object["y1"]),
                          (closest_object["x2"], closest_object["y2"] - 10), (0, 255, 0), 2)
            cv2.rectangle(last_frame_with_closest_object, (closest_object["x1"], closest_object["y1"]),
                          (closest_object["x2"], closest_object["y2"] - 10), (0, 255, 0), 2)

            cv2.putText(img,
                        f'{closest_object["class_name"]} : {closest_object["confidence"]} | {x:.2f},{y:.2f},{z:.2f}',
                        (closest_object["center_x"], closest_object["center_y"] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.putText(last_frame_with_closest_object,
                        f'{closest_object["class_name"]} : {closest_object["confidence"]} | {x:.2f},{y:.2f},{z:.2f}',
                        (closest_object["center_x"], closest_object["center_y"] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            cv2.circle(img, (closest_object["center_x"], closest_object["center_y"]), 5, (0, 0, 255), -1)

            cv2.circle(last_frame_with_closest_object, (closest_object["center_x"], closest_object["center_y"]), 5,
                       (0, 0, 255), -1)

        if processing_paused and last_frame_with_closest_object is not None:
            motor_positions = f'MOTOR,{theta1:.2f},{theta2:.2f},{theta3:.2f},{theta4:.2f},{theta5:.2f}'
            write_to_arduino(motor_positions)
            cv2.imshow("Object Detection", last_frame_with_closest_object)
            if read_from_arduino() == "DONE":
                processing_paused = False
        else:
            cv2.imshow("Object Detection", img)

        # Check if the user wants to pause or quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            processing_paused = not processing_paused

    cap.release()
    cv2.destroyAllWindows()
    detector.stop_camera()  # Stop the Realsense camera


if __name__ == "__main__":
    main()
