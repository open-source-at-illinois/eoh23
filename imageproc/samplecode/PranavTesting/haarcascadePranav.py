import numpy as np
from matplotlib import pyplot as plt
import cv2

def haarcascadePranav(frame, canvas):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cascade files used to detect faces and eyes
    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('../haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('../nose.xml')
    


    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    #detects faces for the image
    # for (x, y, w, h) in faces:
    #     face = frame[y:y+h, x:x+w]

    #     # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) #draws a box around each face
    #     roi_gray = gray_frame[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ex, ey, ew, eh) in nose:  # draws a box around the nose
        #cv2.rectangle(canvas, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.circle(canvas,(ex + ew//2, ey + eh//2),10,(0,0,255), -1)
            # bunnyears_resized = cv2.resize(bunnyears, (w, h))
            # bunnyears_fg = bunnyears_resized[:,:,:3]
        # Convert the face and sunglasses layers to floating point representation
            # face_normalized = face.astype(float)/255.0
            # bunnyears_fg_normalized = bunnyears_fg.astype(float)/255.0
            # face_filtered = cv2.multiply(face_normalized, 1.0 - bunnyears_fg_normalized)
            # face_filtered = cv2.add(face_filtered, bunnyears_fg_normalized)
            # face_filtered = (face_filtered * 255).astype(np.uint8)
            # frame[y:y+h, x:x+w] = face_filtered
            # frame = cv2.add(frame[y:y+h, x:x+h], bunnyears_fg_normalized)




    return frame,nose
