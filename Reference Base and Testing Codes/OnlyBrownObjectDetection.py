import cv2
import numpy as np


def detect_brown_objects(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the brown color in HSV
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])

    # Create a mask to detect brown color in the frame
    brown_mask = cv2.inRange(hsv_frame, lower_brown, upper_brown)

    # Find contours in the mask
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the detected contours
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)

        # Filter out small contours (noise)
        if area > 1000:  # Adjust the threshold as needed
            # Get the bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center of the contour
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw a circle at the center of the detected object
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame


# Open a connection to the webcam (change the index if you have multiple webcams)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect brown objects in the frame
    result_frame = detect_brown_objects(frame)

    # Display the frame with detected objects
    cv2.imshow('Brown Object Detection', result_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
