import cv2
import numpy as np
from matplotlib import pyplot as plt

# Capture video from the webcam
capture = cv2.VideoCapture(0)

#create cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
sunglasses = cv2.imread('sunglasses.png', -1)

while True:
    # Get the current frame
    _, frame = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each face
    for (x, y, w, h) in faces:
        # Crop the face
        face = frame[y:y+h, x:x+w]

        # Resize the sunglasses image to the size of the face
        sunglasses_resized = cv2.resize(sunglasses, (w, h))

        # Split the sunglasses image into its foreground and background layers
        sunglasses_fg = sunglasses_resized[:,:,:3]
        sunglasses_bg = cv2.bitwise_not(sunglasses_resized[:,:,3])

        # Convert the face and sunglasses layers to floating point representation
        face_float = face.astype(float)
        sunglasses_fg_float = sunglasses_fg.astype(float)

        # Normalize the face and sunglasses layers
        face_normalized = face_float / 255.0
        sunglasses_fg_normalized = sunglasses_fg_float / 255.0

        # Multiply the normalized face layer by (1 - alpha)
        face_filtered = cv2.multiply(face_normalized, 1.0 - sunglasses_fg_normalized)

        # Add the filtered face and sunglass layers
        face_filtered = cv2.add(face_filtered, sunglasses_fg_normalized)

        # Convert the face to 8-bit representation
        face_filtered = (face_filtered * 255).astype(np.uint8)

        # Overlay the filtered face on the original frame
        frame[y:y+h, x:x+w] = face_filtered

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
capture.release()
cv2.destroyAllWindows()