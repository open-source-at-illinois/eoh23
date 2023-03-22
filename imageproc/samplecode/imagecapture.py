# import the opencv library
import cv2
from filters import blur, heatmap, adaptThreshold, sobelDerivative, contours, img64
# define a video capture object

vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_EXPOSURE, -1)
mode = 'sobelDerivative'
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #cropping frame
    frame = frame[0:768, 310:1078]

    # Display the resulting frame
    if mode =='def':
        cv2.imshow('frame', frame)
    elif mode == 'blur':
        cv2.imshow('frame', blur(frame))
    elif mode == 'heatmap':
        cv2.imshow('frame', heatmap(frame))
    elif mode == 'adaptiveThreshold':
        cv2.imshow('frame', adaptThreshold(frame))    
    elif mode == 'sobelDerivative':
        cv2.imshow('frame', sobelDerivative(frame))
    elif mode == 'contours':
        cv2.imshow('frame', contours(frame))
    elif mode == '64bit':
        cv2.imshow(mode, img64(frame))


    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()

# converting from 768*768 to 64x64
