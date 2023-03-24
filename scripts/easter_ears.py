import numpy as np
from matplotlib import pyplot as plt
import cv2

def bunnyears(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cascade files used to detect faces and eyes
    face_cascade = cv2.CascadeClassifier(
        "dependencies/haarcascade_frontalface_default.xml")
    bunnyears = cv2.imread('dependencies/bunnyears.png', -1)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # detects faces for the image
    for (x, y, w, h) in faces:
        faceprime = frame[y-h:y, x:x+w]
        bunnyears_resized = cv2.resize(bunnyears, (w, h))

        bunnyears_fg = bunnyears_resized[:, :, :3]
    # # Convert the face and sunglasses layers to floating point representation
        face_normalized = faceprime.astype(float)/255.0
        bunnyears_fg_normalized = bunnyears_fg.astype(float)/255.0
        try:
            face_filtered = cv2.multiply(
                face_normalized, 1 - bunnyears_fg_normalized)  # error
            face_filtered = cv2.add(face_filtered, bunnyears_fg_normalized)
            face_filtered = (face_filtered * 255).astype(np.uint8)
            frame[y-h:y, x:x+w] = face_filtered
        except:
            pass
    return frame


