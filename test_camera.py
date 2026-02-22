# Test this FIRST in new file test_camera.py
import cv2
cap = cv2.VideoCapture(0)
print("Camera 0:", cap.isOpened())
cap = cv2.VideoCapture(1)
print("Camera 1:", cap.isOpened())
cap.release()
