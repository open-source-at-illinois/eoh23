from rgbmatrix import graphics
import time
import cv2

class GraphicsTest(SampleBase):
    def __init__(self, *args, **kwargs):
        super(GraphicsTest, self).__init__(*args, **kwargs)

    def run(self):
        canvas = self.matrix

        vid = cv2.VideoCapture(0) #can be set to 1 (depends on the COM port being used for the webcam
        while True:
            ret, frame = vid.read() #reads the current frame from video feed
            frame = frame[0:768, 310:1078] #frame cropped to a size of x:x, where x is a multiple of 64

            heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)  # grayscale conversion
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # heatmap color application
            
            height, width = frame.shape[:2] #fetches height and width of the frame (x)
            # Desired "pixelated" size
            w, h = (64, 64)
            # Resize input to "pixelated" size
            temp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)             # Initialize output image
            
            frame = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST) # convert frame to a 64x64 array
            
            for i in range(64): # for every row in the frame
                for j in range(64): #for every column  in frame
                    canvas.setPixel(
                        i, j, frame[i][j][0], frame[i][j][1], frame[i][j][2]) #sets R G B values for the ixjth pixel



# Main function
if __name__ == "__main__":
    graphics_test = GraphicsTest()
    if (not graphics_test.process()):
        graphics_test.print_help()
