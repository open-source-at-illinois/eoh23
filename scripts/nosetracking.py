# import the opencv library
import cv2
import numpy as np

def nosetracking(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cascade files used to detect faces and eyes
    face_cascade = cv2.CascadeClassifier(
        "dependencies/haarcascade_frontalface_default.xml")
    # eye_cascade = cv2.CascadeClassifier('../haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('dependencies/nose.xml')

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    try:
        faces = faces[0]
    except:
        pass

    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ex, ey, ew, eh) in nose:  # draws a box around the nose
        # cv2.rectangle(canvas, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # cv2.circle(canvas,(ex + ew//2, ey + eh//2),10,(0,0,255), -1)
        pts = ((ex + ew//2, ey + eh//2))
    return pts


# VERY HELPFUL RESOURCES:
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://www.geeksforgeeks.org/saving-a-video-using-opencv/#
