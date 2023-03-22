# import the opencv library
import cv2
import numpy as np
from collections import deque

# # define a video capture object
vid = cv2.VideoCapture(0)

# # for saving a video
# frame_width = int(vid.get(3))
# frame_height = int(vid.get(4))
   
# size = (frame_width, frame_height)
# result_recorded = cv2.VideoWriter('nose_tracking.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)


MAX_BUFFER = 32



ret, frame = vid.read()
canvas = np.zeros(frame.shape, np.uint8)
pts = deque(maxlen=MAX_BUFFER)
while(True):
    ret, frame = vid.read()
    
    # frame = frame[0:768, 310:1078]
    #cropping frame


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cascade files used to detect faces and eyes
    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
    # eye_cascade = cv2.CascadeClassifier('../haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('../nose.xml')


    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ex, ey, ew, eh) in nose:  # draws a box around the nose
        # cv2.rectangle(canvas, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # cv2.circle(canvas,(ex + ew//2, ey + eh//2),10,(0,0,255), -1)
        pts.appendleft((ex + ew//2, ey + eh//2))
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
		# them
        if pts[i - 1] is None or pts[i] is None:
            continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
        thickness = int(np.sqrt(MAX_BUFFER / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    
    # result_recorded.write(frame)
    cv2.imshow('frame', cv2.flip(frame, 1))
    
    # cv2.imshow('canvas', cv2.flip(canvas, 1))

    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
# result_recorded.release()
cv2.destroyAllWindows()

# VERY HELPFUL RESOURCES:
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://www.geeksforgeeks.org/saving-a-video-using-opencv/#