import GetFace
import numpy as np
import cv2
import sys

class trackFace:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.fontSize=0.75
        self.lineSize=2
        self.fontColor=(255, 255, 255)#white
        self.lineColor=(255, 0, 0)#blue
        self.getFace=GetFace.GetFace()
        self.img=self.getFace.getFirstFram()
       
        
    def getFrame(self):
        self.getFace=GetFace.GetFace()
        self.img=self.getFace.getFirstFram()
        #self.x, self.y, self.w, self.h, self.name=self.getFace.coordinatesCascade()
        self.x, self.y, self.w, self.h, self.name=self.getFace.coordinatesFaceDetection("Alfonso")
        if self.x==0 and self.y==0: self.faceDetected=False
        else: self.faceDetected=True
        
        
    def showFrameCaptured(self):
        if self.faceDetected:
            self.img = cv2.rectangle(self.img,(self.x,self.y),(self.x+self.w,self.y+self.h),self.lineColor,self.lineSize)
            cv2.imshow('img',self.img)
        else: print("Face no detected")
        
    def trackFace(self):
        if self.faceDetected:
            tracker = cv2.TrackerCSRT_create()
            video = cv2.VideoCapture(0)
            ok, frame = video.read()
            if not ok:
                print ('Cannot read video file')
                sys.exit()
            
            bbox = (self.x, self.y, self.w, self.h)
            ok = tracker.init(frame, bbox)
             
            while True:
                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    break
                 
               
                # Update tracker
                ok, bbox = tracker.update(frame)
             
                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, self.lineColor, self.lineSize)
                else :
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), self.font, self.fontSize,self.fontColor,self.lineSize)
             
                # Display tracker type on frame
                cv2.putText(frame, "CSRT Tracker", (100,20), self.font, self.fontSize,self.fontColor,self.lineSize);
                
                cv2.putText(frame, self.name, (int(bbox[0] + 6), int(bbox[3]+bbox[1] - 6)), self.font, self.fontSize,self.fontColor,self.lineSize)
        
                # Display result
                cv2.imshow("Tracking "+self.name, frame)
             
                # Exit if q pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video.release()
                    cv2.destroyAllWindows()
                    break
        else:print("Face no detected")
        
        
        

if __name__=='__main__':
    track= trackFace()
    track.getFrame()
    track.showFrameCaptured()
    track.trackFace()
    
