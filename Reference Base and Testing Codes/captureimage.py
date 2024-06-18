import os

import cv2

# Initialize the camera
camera = cv2.VideoCapture(1)  # 0 represents the default camera (usually the built-in webcam)

os.chdir("./images")
if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

try:
    for i in range(11):
        ret, frame = camera.read()

        if not ret:
            print(f"Error capturing image {i + 1}.")
            break

        # Save the captured image with a unique filename
        image_filename = f"image{i}.jpg"
        cv2.imwrite(image_filename, frame)

        print(f"Image {i + 1} saved as {image_filename}")

        # Show the captured image for a few seconds
        cv2.imshow(f"Captured Image {i + 1}", frame)
        cv2.waitKey(10000)  # Show the image for 2 seconds

finally:
    # Release the camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
