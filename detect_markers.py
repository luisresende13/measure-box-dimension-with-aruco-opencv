import cv2
import numpy as np

# Load the dictionary that was used to generate the markers.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values.
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector with the dictionary and parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Capture the video stream from the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # Draw the markers on the frame
    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the resulting frame
    cv2.imshow('ArUco Marker Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
