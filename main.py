import cv2
import numpy as np
from collections import deque
from scripts import easter_ears, illini_eyes, nosetracking

camera_num = 0 #picks which camera to use. For a device with multiple cameras, camera_num can be set to 1, 2 etc.

vid = cv2.VideoCapture(0) #captures video feed from the selected camera


#defining local variables for nose reading
MAX_BUFFER = 20
pts = deque(maxlen=MAX_BUFFER)




mode = 'nosetracking' #set a mode variable
while (True):
    
    ret, frame = vid.read()  # saves the current frame from the video feed
    canvas = np.zeros(frame.shape, np.uint8)
    frame = cv2.flip(frame, 1)

    #function call
    if mode =='easter_ears':
        cv2.imshow('frame',  easter_ears.bunnyears(frame))
    elif mode == 'illini_eyes':
        try:
            cv2.imshow('frame',  illini_eyes.IlliniI(frame))
        except:
            continue
    elif mode =='nosetracking':
        try:
            pts.appendleft(nosetracking.nosetracking(frame))
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(MAX_BUFFER / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                cv2.imshow('frame',  frame)
        except:
            continue
    # the 'q' button is set as the quit button
    else:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
