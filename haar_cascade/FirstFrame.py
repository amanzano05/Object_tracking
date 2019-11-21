import cv2
import numpy as np


webcam = cv2.VideoCapture(0)
check, frame = webcam.read()
while not check:
    check, frame = webcam.read()

webcam.release()
cv2.imshow("Frame", frame)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
 

