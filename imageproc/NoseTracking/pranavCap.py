# import the opencv library
import cv2
import numpy as np
from collections import deque
import sys

LINE_LENGTH = 32
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
OUTLIER_BUFFER = 500
MAX_NOSES = 1 if len(sys.argv) < 2 else int(sys.argv[1])
NOSE_XML = 'nose.xml'


# OpenCV initialization (start camera, initialize deque for each nose, select )
vid = cv2.VideoCapture(0)
pts = []
for i in range(MAX_NOSES):
    pts.append(deque(maxlen=LINE_LENGTH))
nose_cascade = cv2.CascadeClassifier(NOSE_XML)



while(True):
    ret, frame = vid.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # match each nose detected (as a cartesian point) to a deque
    for i,(ex, ey, ew, eh) in enumerate(nose):  
        if i >= MAX_NOSES: 
            break

        curX = ex + ew//2
        curY = ey + eh//2

        if not pts[i]:
            pts[i].appendleft((curX, curY))
        else:
            for j in range(MAX_NOSES):
                last_x, last_y = pts[j][-1]
                if abs(last_x - curX) < OUTLIER_BUFFER and abs(last_y - curY) < OUTLIER_BUFFER:
                    pts[j].appendleft((curX, curY))
                    break

    # draw lines for each deque (nose trail)
    for j,pt in enumerate(pts):
        color = RED
        if j % 3 == 1:
            color = GREEN
        elif j % 3 == 2:
            color = BLUE
        for i in range(1, len(pt)):

            # if either of the tracked points are None, ignore
            if pt[i - 1] is None or pt[i] is None:
                continue

            thickness = int(np.sqrt(LINE_LENGTH / float(i + 1)) * 2.5)
            cv2.line(frame, pt[i - 1], pt[i], color, thickness)
    
    cv2.imshow('frame', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()

cv2.destroyAllWindows()


# Goals:

# VERY HELPFUL RESOURCES:
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://www.geeksforgeeks.org/saving-a-video-using-opencv/#