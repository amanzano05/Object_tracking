import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
check, frame = webcam.read()
print(check) #prints true as long as the webcam is running
print(frame) #prints matrix values of each framecd 
cv2.imshow("Capturing", frame)

webcam = cv2.VideoCapture(0)
while True:
     
    check, frame = webcam.read()
 
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): 
        break

webcam.release()
print("Camera off.")
print("Program ended.")
cv2.destroyAllWindows()
        

