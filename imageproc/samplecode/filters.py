import cv2
import numpy
import matplotlib.pyplot as plot


def blur(frame):
    #applying a bilateral filter 
    #params are the frame, diam of pixel neighborhood, sigma space and border type
    frame = cv2.bilateralFilter(frame, 20, 115, 35)    
    return frame

def heatmap(frame):
    heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_BONE) #grayscale conversion
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #heatmap applied to b/w frame
    return heatmap

def shades(frame):
    return None

def adaptThreshold(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 7)
    return thresh2


#Contours

def sobelDerivative(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame, 25, 180, apertureSize = 3)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    return edge

def contours(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(frame, (5,5))
    return cv2.Laplacian(frame, cv2.CV_64F)


def img64(frame):

    # Get input size
    height, width = frame.shape[:2]
    # Desired "pixelated" size
    w, h = (64, 64)
    # Resize input to "pixelated" size
    temp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output
