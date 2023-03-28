import cv2
import numpy as np
from collections import deque
from scripts import easter_ears, illini_eyes, nosetracking
import sys, os

camera_num = 0 #picks which camera to use. For a device with multiple cameras, camera_num can be set to 1, 2 etc.

vid = cv2.VideoCapture(0) #captures video feed from the selected camera


#defining local variables for nose reading
MAX_BUFFER = 20
pts = deque(maxlen=MAX_BUFFER)



nose_cascade = cv2.CascadeClassifier('dependencies/nose.xml') #pranav's REFACTOR: only done once, no need to keep in while loop

mode = 'def' #set a mode variable
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
        # pts.appendleft(nosetracking.nosetracking(frame, pts, nose_cascade))
        nosetracking.nosetracking(frame, pts, nose_cascade)
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            if pts[i - 1] is None or pts[i] is None:
                continue

            if pts[i-1] and pts[i]:
                thickness = int(np.sqrt(MAX_BUFFER / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            cv2.imshow('frame',  frame)
        # try:
        #     pts.appendleft(nosetracking.nosetracking(frame, pts, nose_cascade))
        #     for i in range(1, len(pts)):
        #         # if either of the tracked points are None, ignore
        #         if pts[i - 1] is None or pts[i] is None:
        #             continue

        #         thickness = int(np.sqrt(MAX_BUFFER / float(i + 1)) * 2.5)
        #         cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        #         cv2.imshow('frame',  frame)
        # except Exception as err:
        #     print(repr(err))
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(str(exc_tb.tb_lineno))
        #     sys.exit()
    # the 'q' button is set as the quit button
    else:
        cv2.imshow('frame', frame)
    
    ### code to switch between different modes
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('i'):
        mode = 'illini_eyes'  
    elif key == ord('b'):
        mode = 'easter_ears'
    elif key == ord('n'):
        mode = 'nosetracking'
    elif key == ord('d'):
        mode = 'def'
    else:
        pass
            
vid.release()
cv2.destroyAllWindows()
