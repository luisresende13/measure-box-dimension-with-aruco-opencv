import cv2
from object_detector import *
import numpy as np
import time

def increase_contrast(frame, alpha=1.5, beta=50):
    """
    Increase the contrast of the frame.
    :param frame: input frame
    :param alpha: contrast control (1.0-3.0)
    :param beta: brightness control (0-100)
    :return: frame with increased contrast
    """
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha)
    return adjusted


# Load Aruco detector
# parameters = cv2.aruco.DetectorParameters_create()
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


# Load Object Detector
detector = HomogeneousBgDetector()

# Load Cap
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

n_frames = 0
fps = 0.00
t = time.time()

while True:
    _, img = cap.read()

    img = increase_contrast(img, alpha=2.0)

    n_frames += 1

    if n_frames % 30 == 0:
        fps = round(n_frames / (time.time() - t), 2)
        t = time.time()
        n_frames = 0

    cv2.putText(img, f"FPS: {fps}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    # Get Aruco marker
    # corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    corners, _, _ = aruco_detector.detectMarkers(img)
    if corners:

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)



    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()