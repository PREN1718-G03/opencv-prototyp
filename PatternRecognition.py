import cv2
import copy
import time
from VideoStream import VideoStream

# Comment these two lines if you want to use the internal camera
# cam  = VideoStream().start()
# Let the camera sensor warm up
# time.sleep(2.0)

# Uncomment this line to use the internal camera
cam = cv2.VideoCapture(0)



def recogniseTarget():
    pass

while True:
    # If local camera uncomment this
    ret, frame = cam.read()
    # frame = cam.read()

    cv2.imshow('Frame', frame)

    # Change the picture to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Gray', gray)

    # Calculate the binary threshold -> Everything lower than threshold turns black, everything over turns white
    retthresh, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Show the calculated threshold
    cv2.imshow('Threshold', threshold)

    # I don't want a reference to the frame object, I want a copy
    cntImage = copy.copy(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Show the recognised rectangles
cam.release()
cv2.destroyAllWindows()