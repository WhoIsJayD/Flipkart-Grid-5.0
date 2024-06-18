import cv2

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed (e.g., 'XVID', 'MJPG', 'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Change the filename and frame size as needed

# Open the default camera (usually 0) or specify a different camera index
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the OpenCV window
cap.release()
out.release()
cv2.destroyAllWindows()
