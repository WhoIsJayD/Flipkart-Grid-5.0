import math
import time
from queue import Queue
from threading import Event

import cv2
import numpy

from arduino_communication import ArduinoComm
from esp32_cam_thread import ObjectDetectionThread
from object_detector import ObjectDetector


def inverse_kinematics(x, z, y):
    # Defining Link Lengths
    d1 = 8.4
    d2 = 54.2
    d3 = 51.7
    d4 = 7
    d5 = 25.5

    # Defining Rotation Matrix
    nx, ny, nz = 0, -1, 0
    sx, sy, sz = 1, 0, 0
    ax, ay, az = 0, 0, 1

    # Defining Angle Constraints Function
    def angle_constraint(angle):
        if 0 > angle > -90:
            angle = numpy.abs(angle)
        if -90 > angle > -180:
            angle = numpy.abs(angle) - 90
        if angle < -180:
            angle = numpy.abs(angle) - 180
        if 0 < angle < 90:
            angle = angle
        if 90 < angle < 180:
            angle = angle - 90
        if angle > 180:
            angle = angle - 180
        return angle

    # All Inverse Kinematics Equations
    q1 = math.atan2(y, x)

    eq1 = -(ax * math.cos(q1) + ay * math.sin(q1))
    eq2 = -az

    q5 = math.atan2((nx * math.sin(q1) - ny * math.cos(q1)), (sx * math.sin(q1) - sy * math.cos(q1)))

    c = (x / math.cos(q1)) - (d5 * eq1) - (d4 * eq2)
    d = (d1 - (d4 * eq1) - (d5 * eq2) - z)

    R = (c * c + d * d - d3 * d3 - d2 * d2) / (2 * (d3 * d2))
    t = math.sqrt(1 - R * R)

    q3 = math.atan2(t, R)

    r = d3 * math.cos(q3) + d2
    s = d3 * math.sin(q3)

    q2 = math.atan2((r * d) - (s * c), (r * c) + (s * d))

    eq3 = math.atan2(-(ax * math.cos(q1) + ay * math.sin(q1)), -az)
    q4 = eq3 - (q2 + q3)

    # Converting angles in radians to degrees
    angle1 = q1 * (180 / math.pi)
    angle2 = q2 * (180 / math.pi)
    angle3 = q3 * (180 / math.pi)
    angle4 = q4 * (180 / math.pi)
    angle5 = q5 * (180 / math.pi)

    angle1 = angle_constraint(angle1)
    angle2 = angle_constraint(angle2)
    angle3 = angle_constraint(angle3)
    angle4 = abs(90 - angle2 - angle3)
    angle5 = 90

    return angle1, angle2, angle3, angle4, angle5


class State:
    INITIALIZING = "Initializing"
    PICKING_UP = "Picking Up"
    PROCESSING_PAUSED = "Processing Paused"
    PICKED_UP = "Picked Up"
    PICK_UP_PICKED = "Picked Up Pose"
    DROP_ZONE_PICKED = "Drop Zone Picked"
    ESP_POSE_PICKED = "ESP Pose Picked"


def perform_object_detection(model_path=r"G:\Grid5.0\Detection Models\QRCode.pt", confidence_threshold=0.7,
                             detect_every=3, max_duration=5):
    video_sources = ["http://192.168.137.122:81/", "http://192.168.137.241:81/"]
    stop_event = Event()
    result_queue = Queue()
    threads = []

    # Record start time
    start_time = time.time()

    for i, video_source in enumerate(video_sources, start=1):
        thread = ObjectDetectionThread(i, model_path, video_source, stop_event, result_queue, confidence_threshold,
                                       detect_every)
        threads.append(thread)
        thread.start()

    # Wait for the specified duration or until a QR code is detected
    while time.time() - start_time < max_duration and not any(thread.is_qr_detected() for thread in threads):
        time.sleep(1)  # Sleep for 1 second to avoid busy waiting

    # Set stop_event to stop all threads
    stop_event.set()

    for thread in threads:
        thread.join()

    print("Stopping all threads as QR code detected or time threshold reached.")

    detection_results = []
    while not result_queue.empty():
        detection_data = result_queue.get()
        detection_results.append(detection_data)

    return detection_results


def write_motor_poses(arduino_comm, motor_positions):
    # return True
    arduino_comm.write_data(motor_positions)
    if arduino_comm.read_data() == "DONE":
        print(f"Written the motor pose: {motor_positions}")
        return True


def main():
    global motor_positions
    model_path = r"G:\Grid5.0\Detection Models\BoxDetection.pt"
    class_names = ["box"]
    confidence_threshold = 0.80
    poses = {"INITIALIZING": "MOTOR,0,0,0,0,0,0,0", "PICKING_UP": "MOTOR,95,10,60,0,98,0,0",
             "DROP_ZONE_PICKED": 'MOTOR,0,15,40,10,90,0,0', 'ESP_POSE_PICKED_1': 'MOTOR,0,10,40,10,90,0,0',
             'ESP_POSE_PICKED_2': 'MOTOR,0,10,40,10,90,90,0',
             "DROP_IT": "MOTOR,0,20,50,10,90,0,1"}  # Don't change the last two values of the poses

    cap = cv2.VideoCapture(1)
    arduino_comm = ArduinoComm()

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

        if not success:
            break

        print(f"Current State: {state}")

        if state == State.INITIALIZING:
            print("Entering INITIALIZING state...")
            initial_pose = write_motor_poses(arduino_comm, poses["INITIALIZING"])
            # print(initial_pose)
            if initial_pose:
                state = State.PICKING_UP
                time.sleep(1)

        elif state == State.PICKING_UP:
            print("Entering PICKING_UP state...")
            pick_up = write_motor_poses(arduino_comm, poses["PICKING_UP"])
            if pick_up:
                state = State.PROCESSING_PAUSED
                time.sleep(1)

        elif state == State.PROCESSING_PAUSED:
            print("Entering PROCESSING_PAUSED state...")

            closest_object = None
            min_distance = float('inf')

            for obj in detected_objects:
                if 0 < obj["z3d"] < min_distance:
                    min_distance = obj["z3d"]
                    closest_object = obj
                    print(closest_object)

            if closest_object and closest_object["z3d"] > 0:
                state = State.PICKED_UP
                time.sleep(1)
                # last_frame_with_closest_object = img.copy()

        elif state == State.PICKED_UP:
            print("Entering PICKED_UP state...")
            x, y, z = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
            # inverse_angles = (x, y, z)
            theta1, theta2, theta3, theta4, theta5 = inverse_kinematics(x, y, z)
            print(theta1, theta2, theta3, theta4, theta5)

            motor_positions = f'MOTOR,{theta1:.2f},{(theta2 + 20):.2f},{theta3:.2f},{theta4:.2f},{theta5:.2f},0,0'
            pick_up = write_motor_poses(arduino_comm, motor_positions)
            time.sleep(9)
            if pick_up:
                # cv2.imshow("Object Detection", last_frame_with_closest_object)
                state = State.PICK_UP_PICKED
                time.sleep(1)

        elif state == State.PICK_UP_PICKED:
            print("Entering PICK_UP_PICKED state...")
            motor_positions_pick = motor_positions.split(",")
            motor_positions_pick[2] = '0'
            motor_positions_pick[3] = '70'
            print(motor_positions_pick)
            motor_positions_pick = f"MOTOR,{motor_positions_pick[1]}, {motor_positions_pick[2]}, {motor_positions_pick[3]}, {motor_positions_pick[4]},{motor_positions_pick[5]},{motor_positions_pick[6]},{motor_positions_pick[7]}"
            print(motor_positions_pick)

            pick_up_pose_picked = write_motor_poses(arduino_comm, motor_positions_pick)
            time.sleep(5)
            if pick_up_pose_picked:
                state = State.DROP_ZONE_PICKED
                time.sleep(1)

        elif state == State.DROP_ZONE_PICKED:
            print("Entering DROP_ZONE_PICKED state...")
            drop_zone_picked = write_motor_poses(arduino_comm, poses["DROP_ZONE_PICKED"])
            time.sleep(5)
            if drop_zone_picked:
                state = State.ESP_POSE_PICKED
                time.sleep(1)

        elif state == State.ESP_POSE_PICKED:
            write_motor_poses(arduino_comm, poses["DROP_IT"])
            time.sleep(5)
            print("Just Move Down")
            # already = False
            # print("Entering ESP_POSE_PICKED state...")
            # esp_pose1_picked = write_motor_poses(arduino_comm, poses["ESP_POSE_PICKED_1"])
            # if esp_pose1_picked:
            #     results = perform_object_detection()
            #     if results and results[0]['thread_index'] == 1:
            #         already = True
            #         esp_1_pose = poses["ESP_POSE_PICKED_1"].split(",")
            #         esp_1_pose[5] = '0'
            #         esp_1_pose[7] = '1'
            #         esp_1_pose = ",".join(esp_1_pose)
            #         write_motor_poses(arduino_comm, esp_1_pose)
            #         print("ESP 1")
            #     elif results and results[0]['thread_index'] == 2:
            #         already = True
            #         esp_2_pose = poses["ESP_POSE_PICKED_1"].split(",")
            #         esp_2_pose[5] = '180'
            #         esp_2_pose[7] = '1'
            #         esp_2_pose = ",".join(esp_2_pose)
            #         write_motor_poses(arduino_comm, esp_2_pose)
            #         print("ESP 2")
            # if not already:
            #     esp_pose2_picked = write_motor_poses(arduino_comm, poses["ESP_POSE_PICKED_2"])
            #     if esp_pose2_picked:
            #         results = perform_object_detection()
            #         if results and results[0]['thread_index'] == 1:
            #             already = True
            #             esp_1_pose = poses["ESP_POSE_PICKED_2"].split(",")
            #             esp_1_pose[5] = '0'
            #             esp_1_pose[7] = '1'
            #             esp_1_pose = ",".join(esp_1_pose)
            #             write_motor_poses(arduino_comm, esp_1_pose)
            #             print("ESP 1")
            #         elif results and results[0]['thread_index'] == 2:
            #             already = True
            #             esp_2_pose = poses["ESP_POSE_PICKED_2"].split(",")
            #             esp_2_pose[5] = '180'
            #             esp_2_pose[7] = '1'
            #             esp_2_pose = ",".join(esp_2_pose)
            #             write_motor_poses(arduino_comm, esp_2_pose)
            #             print("ESP 2")
            # else:
            #     write_motor_poses(arduino_comm, poses["DROP_IT"])
            #     print("Just Move Down")

            state = State.PICKING_UP
            time.sleep(5)

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
