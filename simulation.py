import math
import time
from queue import Queue
from threading import Event

import cv2

from arduino_communication import ArduinoComm
from esp32_cam_thread import ObjectDetectionThread
from object_detector import ObjectDetector

# Link lengths
d1 = 10
d2 = 55
d3 = 52
d4 = 7
d5 = 24


def inverse_kinematics(x, y, z):
    # Calculate joint angles
    q1 = math.atan2(y, x)

    c = x - d4 * math.cos(q1)
    d = y - d4 * math.sin(q1)

    R = (c ** 2 + d ** 2 - d3 ** 2 - d2 ** 2) / (2 * d2 * d3)
    t = math.sqrt(1 - R ** 2)
    q3 = math.atan2(t, R)

    r = d2 + d3 * math.cos(q3)
    s = d3 * math.sin(q3)
    q2 = math.atan2(r * d - s * c, r * c + s * d)

    q4 = math.atan2(z, math.sqrt(x ** 2 + y ** 2)) - q2 - q3
    q5 = 0

    # Convert to degrees
    angle1 = math.degrees(q1)
    angle2 = math.degrees(q2)
    angle3 = math.degrees(q3)
    angle4 = math.degrees(q4)
    angle5 = math.degrees(q5)

    # Apply angle constraints
    angle1 = constrain_angle(angle1)
    angle2 = constrain_angle(angle2)
    angle4 = constrain_angle(angle4)
    angle3 = 90 - angle2 - angle4
    angle5 = constrain_angle(angle5)

    print("Inverse angles : " + str(angle1) + " degrees" + str(angle2) + " degrees" + str(angle3) + " degrees" + str(
        angle4) + " degrees" + str(angle5) + " degrees")
    return angle1, angle2, angle3, angle4, angle5


def constrain_angle(angle):
    if 0 > angle > -90:
        angle = angle + 90
    if 0 > angle < -90:
        angle = abs(angle) + 90
    if 90 < angle < 180:
        angle = angle - 90
    if 90 < angle > 180:
        angle = angle - 180  # or 270 - angle1
    return angle


class State:
    INITIALIZING = "Initializing"
    PICKING_UP = "Picking Up"
    PROCESSING_PAUSED = "Processing Paused"
    PICKED_UP = "Picked Up"
    PICK_UP_PICKED = "Picked Up Pose"
    DROP_ZONE_PICKED = "Drop Zone Picked"
    ESP_POSE_PICKED = "ESP Pose Picked"


def perform_object_detection(model_path=r"G:\Grid5.0\Detection Models\QRCode.pt",
                             confidence_threshold=0.7, detect_every=3):
    video_sources = ["http://192.168.137.122:81/", "http://192.168.137.241:81/"]
    stop_event = Event()
    result_queue = Queue()
    threads = []

    for i, video_source in enumerate(video_sources, start=1):
        thread = ObjectDetectionThread(i, model_path, video_source, stop_event, result_queue, confidence_threshold,
                                       detect_every)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Stopping all threads as QR code detected.")

    detection_results = []
    while not result_queue.empty():
        detection_data = result_queue.get()
        detection_results.append(detection_data)

    return detection_results


def write_motor_poses(arduino_comm, motor_positions):
    arduino_comm.write_data(motor_positions)
    if arduino_comm.read_data() == "DONE":
        print(f"Written the motor pose: {motor_positions}")
        return True


def main():
    model_path = "G:\Grid5.0\Detection Models\BoxDetection.pt"
    class_names = ["box"]
    confidence_threshold = 0.80

    cap = cv2.VideoCapture(1)
    arduino_comm = ArduinoComm()
    poses = {"INITIALIZING": "MOTOR,0,0,0,0,0,0,0", "PICKING_UP": "MOTOR,95,10,60,0,98,0,0",
             "DROP_ZONE_PICKED": 'MOTOR,0,10,40,10,90,0,0', 'ESP_POSE_PICKED': 'MOTOR,0,10,40,10,90,0,0',
             "DROP_IT": "MOTOR,0,10,40,10,90,0,0", "ESP_POSE_PICKED_MOVE_M5_0": 'MOTOR,0,10,40,10,0,0,0',
             "ESP_POSE_PICKED_MOVE_M5_180": 'MOTOR,0,10,40,10,180,0,0', "RELEASE_SUCTION_E1": 'MOTOR,0,10,40,10,0,0,1',
             "RELEASE_SUCTION_E2": 'MOTOR,0,10,40,10,180,0,1', "RELEASE_SUCTION": 'MOTOR,0,10,40,10,90,0,1'}

    object_detector = ObjectDetector(model_path, class_names, confidence_threshold)
    print("Starting object detection...")
    object_detector.start_camera()
    print("Object Detection Started...")

    state = State.INITIALIZING
    print("State Initiated...")
    closest_object = None
    last_frame_with_closest_object = None

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (320, 240))
        detected_objects = object_detector.detect_objects(img)
        time.sleep(4)

        if not success:
            break

        print(f"Current State: {state}")

        if state == State.INITIALIZING:
            print("Entering INITIALIZING state...")
            initial_pose = write_motor_poses(arduino_comm, poses["INITIALIZING"])
            print(initial_pose)
            if initial_pose:
                state = State.PICKING_UP
                time.sleep(0.2)

        elif state == State.PICKING_UP:
            print("Entering PICKING_UP state...")
            pick_up = write_motor_poses(arduino_comm, poses["PICKING_UP"])
            if pick_up:
                state = State.PROCESSING_PAUSED
                time.sleep(0.2)

        elif state == State.PROCESSING_PAUSED:
            print("Entering PROCESSING_PAUSED state...")
            jugad_pose = f"MOTOR,{0},{0},{0},{0},{0},{0}"
            processing_paused = write_motor_poses(arduino_comm, jugad_pose)
            if processing_paused:
                state = State.PICK_UP_PICKED
                time.sleep(0.2)


        elif state == State.PICK_UP_PICKED:
            print("Entering PICK_UP_PICKED state...")
            pick_up_pose_picked = write_motor_poses(arduino_comm, poses["PICKING_UP"])
            if pick_up_pose_picked:
                state = State.DROP_ZONE_PICKED
                time.sleep(0.2)

        elif state == State.DROP_ZONE_PICKED:
            print("Entering DROP_ZONE_PICKED state...")
            drop_zone_picked = write_motor_poses(arduino_comm, poses["DROP_ZONE_PICKED"])
            if drop_zone_picked:
                state = State.ESP_POSE_PICKED
                time.sleep(0.2)

        elif state == State.ESP_POSE_PICKED:
            print("Entering ESP_POSE_PICKED state...")
            esp_pose_picked = write_motor_poses(arduino_comm, poses["ESP_POSE_PICKED"])
            if esp_pose_picked:
                results = perform_object_detection()
                if results and results[0]['thread_index'] == 1:
                    write_motor_poses(arduino_comm, poses["ESP_POSE_PICKED_MOVE_M5_0"])

                    write_motor_poses(arduino_comm, poses["RELEASE_SUCTION_E1"])
                    print("ESP 1")
                elif results and results[0]['thread_index'] == 2:
                    write_motor_poses(arduino_comm, poses["ESP_POSE_PICKED_MOVE_M5_180"])

                    write_motor_poses(arduino_comm, poses["RELEASE_SUCTION_E2"])
                    print("ESP 2")
                else:
                    write_motor_poses(arduino_comm, poses["DROP_IT"])

                    write_motor_poses(arduino_comm, poses["RELEASE_SUCTION"])
                    print("Just Move Down")
                state = State.PICKING_UP
                time.sleep(0.2)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            state = State.PROCESSING_PAUSED if state != State.PROCESSING_PAUSED else State.PICKED_UP

    cap.release()
    cv2.destroyAllWindows()
    object_detector.stop_camera()
    arduino_comm.close()


if __name__ == "__main__":
    main()
