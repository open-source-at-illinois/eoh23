import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# Create cascades
face_cascade = cv2.CascadeClassifier("dependencies/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('dependencies/haarcascade_eye.xml')

# PNG
illiniI = cv2.imread('dependencies/illiniI.png', -1)

def IlliniI(frame):

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each face
    for (x, y, w, h) in faces:
        # Crop the face
        face = frame[y:y+h, x:x+w]

        # Resize
        illiniI_resized = cv2.resize(illiniI, (w, h))

        # Foreground and background layer
        illiniI_fg = illiniI_resized[:, :, :3]
        illiniI_bg = cv2.bitwise_not(illiniI_resized[:, :, 3])
        # Convert to FP rep
        face_float = face.astype(float)
        illiniI_fg_float = illiniI_fg.astype(float)
        # Normalize layers
        try:
            face_normalized = face_float / 255.0
            illiniI_fg_normalized = illiniI_fg_float / 255.0

            # Multiply by (1 - alpha)
            face_filtered = cv2.multiply(
                face_normalized, 1.0 - illiniI_fg_normalized)

            # Add layers
            face_filtered = (face_filtered * 255).astype(np.uint8)

            # Overlay onto original frame
            frame[y:y+h, x:x+w] = face_filtered
            return frame
        except:
            pass
        # Currently not working for Individual eyes, uncomment code
        # Breaking down face
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]

        # Individual Eyes
        # # Find eyes
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     # Crop the eyes
        #     eye = frame[ey:ey+eh, ex:ex+eh]

        #     # Resize illini image to size of the eyes
        #     illiniI_resized = cv2.resize(illiniI, (ew, eh))

        #     #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #     # FG and BG layers
        #     illiniI_fg = illiniI_resized[:,:,:3]
        #     illiniI_bg = cv2.bitwise_not(illiniI_resized[:,:,3])

        #     # Convert eye and Illini layers to floating representation
        #     eye_float = eye.astype(float)
        #     illiniI_fg_float = illiniI_fg.astype(float)

        #     # Normalize layers
        #     eye_normalized = eye_float / 255.0
        #     illiniI_fg_normalized = illiniI_fg_float / 255.0

        #     # Mutiply normalized eye layer by (1 - alpha)
        #     eye_filtered = cv2.multiply(eye_normalized, 1.0 - illiniI_fg_normalized)

        #     # Add filtered eye and Illini layers
        #     eye_filtered = cv2.add(eye_filtered, illiniI_fg_normalized)

        #     # Convert the eye to 8-bit representation
        #     eye_filtered = (eye_filtered * 255).astype(np.uint8)

        #     #Overlay the filtered eyes on the original frame
        #     frame[y+ey:y+ey+eh, x+ex:x+ex+ew] = eye_filtered
