# import the opencv library
import cv2
import numpy as np



def nosetracking(frame, pts, nose_cascade):
    SPIKE_TOLERANCE = 400
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cascade files used to detect nose

    nose = nose_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (ex, ey, ew, eh) in nose:  # draws a box around the nose
        # cv2.rectangle(canvas, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # cv2.circle(canvas,(ex + ew//2, ey + eh//2),10,(0,0,255), -1)
        curX = ex + ew//2
        curY = ey + eh//2
        if not pts:
            pts.appendleft((curX, curY))
            break
        lastX, lastY = pts[-1]
        if abs(lastX - curX) < SPIKE_TOLERANCE and abs(lastY - curY) < SPIKE_TOLERANCE:
            pts.appendleft((curX, curY))
        break
    # return pts


# VERY HELPFUL RESOURCES:
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://www.geeksforgeeks.org/saving-a-video-using-opencv/#
