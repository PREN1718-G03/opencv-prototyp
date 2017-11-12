# Imports
import cv2
import time
import numpy as np
from VideoStream import VideoStream

vs = VideoStream().start()
time.sleep(2.0)

# Edges of the picture frame, lower left, lower right, upper right, upper left
frameEdges = np.array(
    [
        [0.0,0.0],
        [0.0, 480.0],
        [480.0, 640.0],
        [640.0, 0.0]
    ]
)
edgePoints = np.array(frameEdges, np.float32)

while True:
    # Capture frame-by-frame
    frame = vs.read()
    RECT_FOUND = False

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 1, 10, 120)

    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.drawContours(frame, contours, -1, (0,250,0), 2)
    # Display the resulting frame
    cv2.imshow('Video', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()