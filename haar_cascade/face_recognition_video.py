import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
videoCapture=cv2.VideoCapture(0) #"/dev/bus/usb/001/002"
scale_factor=1.05
if not (videoCapture.isOpened()):
    print("Could not open video device")
else:
    while 1:
        ret, pic =videoCapture.read()
        faces=faceCascade.detectMultiScale(pic,scale_factor,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(pic,(x,y), (x+w, y+h),(255,0,0),2)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(pic, 'Me',(x,y),font,2,(255,255,255),2,cv2.LINE_AA)
    
        print("Number of faces found {} ".format(len(faces)))
        cv2.imshow('face', pic)
        k=cv2.waitKey(27) & 0xff
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
videoCapture.release()
cv2.destroyAllWindows()