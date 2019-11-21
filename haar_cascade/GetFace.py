#!/usr/bin/env python
import cv2 
import numpy as np
import face_recognition

class GetFace:
    def __init__(self):
        self.x=0
        self.y=0
        self.w=0
        self.h=0
    
        
    def getFirstFram(self):
        video=cv2.VideoCapture(0)
        validFrame, self.frame = video.read()
        while not validFrame:
            validFrame, self.frame = video.read()

        video.release()
        return self.frame
    
    
    def coordinatesCascade(self):
        name="Unknown"
        faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 
                                      scaleFactor=1.05,
                                      minNeighbors=5,
                                      minSize=(30,30),
                                      flags = cv2.CASCADE_SCALE_IMAGE)
        if len(faces)>0:
            self.x, self.y, self.w, self.h=faces[0]
            
            
        return self.x,self.y,self.w,self.h, name
    
    
    def coordinatesFaceDetection(self, Name):
        # Load a sample picture and learn how to recognize it.
        poncho_image = face_recognition.load_image_file("Poncho.jpg")
        poncho_face_encoding = face_recognition.face_encodings(poncho_image)[0]
        
        # Load a second sample picture and learn how to recognize it.
        mateo_image = face_recognition.load_image_file("Mateo.jpg")
        mateo_face_encoding = face_recognition.face_encodings(mateo_image)[0]
        
        mark_image = face_recognition.load_image_file("Mark.jpg")
        mark_face_encoding = face_recognition.face_encodings(mark_image)[0]
        
        brandon_image = face_recognition.load_image_file("Brandon.jpg")
        brandon_face_encoding = face_recognition.face_encodings(brandon_image)[0]
        
        # Create arrays of known face encodings and their names
        known_face_encodings = [
            poncho_face_encoding,
            mateo_face_encoding,
            mark_face_encoding,
            brandon_face_encoding
        ]
        known_face_names = [
            "Alfonso",
            "Mateo",
            "Mark",
            "Brandon"
            
        ]
        
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
     
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
             
            if name==Name:
                return left, top, right-left, bottom-top, name
        
        return 0,0,0,0, "Unknown"
            
        
            
            
        
    
        
        
        
  
        
        
        
    