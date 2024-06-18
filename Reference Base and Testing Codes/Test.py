from queue import Queue
from threading import Event

import cv2

from arduino_communication import ArduinoComm
from esp32_cam_thread import ObjectDetectionThread
from inverse_kinematics import InverseKinematics
from object_detector import ObjectDetector


def perform_object_detection(model_path=r"G:\Grid5.0\Detection Models\QRCode.pt",
                             confidence_threshold=0.7, detect_every=3):
    video_sources = ["http://192.168.137.209:81/", "http://192.168.137.23:81/"]
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


def write_motor_poses(theta1, theta2, theta3, theta4, theta5):
    arduino_comm.write_data(motor_positions)
    if arduino_comm.read_data() == "DONE":
        print(f"Written the motor pose : {theta1}, {theta2}, {theta3}, {theta4}, {theta5} ")
        return True


def main():
    model_path = "model.pt"
    class_names = ["box"]
    confidence_threshold = 0.90

    cap = cv2.VideoCapture(1)
    arduino_comm = ArduinoComm()

    object_detector = ObjectDetector(model_path, class_names, confidence_threshold)
    object_detector.start_camera()

    processing_paused = False
    last_frame_with_closest_object = None

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1024, 768))

        if not success:
            break

        if not processing_paused:
            initial_pose = write_motor_poses(0, 0, 0, 0, 0)
            if initial_pose:
                pick_up = write_motor_poses(90, 0, 0, 0, 0)
                if pick_up:

                    detected_objects = object_detector.detect_objects(img)

                    closest_object = None
                    min_distance = float('inf')

                    for obj in detected_objects:
                        if 0 < obj["z3d"] < min_distance:
                            min_distance = obj["z3d"]
                            closest_object = obj

                    if closest_object and closest_object["z3d"] > 0:
                        processing_paused = True
                        last_frame_with_closest_object = img.copy()

                        x, y, z = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
                        inverse_angles = InverseKinematics.calculate_inverse_kinematics(x, y, z)
                        theta1, theta2, theta3, theta4, theta5 = inverse_angles

                        if closest_object:
                            x, y, z = closest_object["x3d"], closest_object["y3d"], closest_object["z3d"]
                            cv2.rectangle(img, (closest_object["x1"], closest_object["y1"]),
                                          (closest_object["x2"], closest_object["y2"] - 10), (0, 255, 0), 2)
                            cv2.putText(img,
                                        f'{closest_object["class_name"]} : {closest_object["confidence"]} | {x:.2f},{y:.2f},{z:.2f}',
                                        (closest_object["center_x"], closest_object["center_y"] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                            cv2.circle(img, (closest_object["center_x"], closest_object["center_y"]), 5, (0, 0, 255),
                                       -1)

                if processing_paused and last_frame_with_closest_object is not None:
                    motor_positions = f'MOTOR,{theta1:.2f},{theta2:.2f},{theta3:.2f},{theta4:.2f},{theta5:.2f}'
                    arduino_comm.write_data(motor_positions)
                    cv2.imshow("Object Detection", last_frame_with_closest_object)

                    if arduino_comm.read_data() == "DONE":
                        pick_up_pose_picked = write_motor_poses(0, 0, 0, 0, 0)
                        if pick_up_pose_picked:
                            dropzone_pose_picked = write_motor_poses(0, 0, 0, 0, 0)
                            if dropzone_pose_picked:
                                esp_pose_picked = write_motor_poses(0, 0, 0, 0, 0)
                                if esp_pose_picked:

                                    results = perform_object_detection()
                                    if results[0]['thread_index'] == 1:
                                        print("ESP 1")
                                    elif results[0]['thread_index'] == 2:
                                        print("ESP 2")
                                    else:
                                        print("Just Move Down")

                                    processing_paused = False

                else:
                    cv2.imshow("Object Detection", img)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    processing_paused = not processing_paused

    cap.release()
    cv2.destroyAllWindows()
    object_detector.stop_camera()
    arduino_comm.close()


if __name__ == "__main__":
    main()
