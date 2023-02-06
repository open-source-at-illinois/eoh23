# import the opencv library
import cv2
from filters import blur, heatmap, adaptThreshold, sobelDerivative, contours, img64
# define a video capture object
vid = cv2.VideoCapture(0)

mode = 'blur'
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = frame[0:768, 310:1078]

    #cropping frame

    # Display the resulting frame
    if mode =='def':
        cv2.imshow('frame', img64(frame))
    elif mode == 'blur':
        cv2.imshow('frame', img64(blur(frame)))
    elif mode == 'heatmap':
        cv2.imshow('frame', img64(heatmap(frame)))
    elif mode == 'adaptiveThreshold':
        cv2.imshow('frame', img64(blur(adaptThreshold(frame))))
    elif mode == 'sobelDerivative':
        cv2.imshow('frame', img64(sobelDerivative(frame)))
    elif mode == 'contours':
        cv2.imshow('frame', img64(contours(frame)))


    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

vid.release()
cv2.destroyAllWindows()