# Imports
import cv2
import sys
import time
import os

# Prepare the Raspberry Pi Camera
# https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
class VideoStream:
    def __init__(self, src=0, resolution=(320, 240),
                 framerate=32):
        # check to see if the picamera module should be used
        # only import the picamera packages unless we are
        # explicity told to do so -- this helps remove the
        # requirement of `picamera[array]` from desktops or
        # laptops that still want to use the `imutils` package
        from PiVideoStream import PiVideoStream

        # initialize the picamera stream and allow the camera
        # sensor to warmup
        self.stream = PiVideoStream(resolution=resolution, framerate=framerate)
    def start(self):
        # start a threaded video stream
        return self.stream.start()
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
    def read(self):
        # return the current frame
        return self.stream.read()
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()

vs = VideoStream().start()
time.sleep(2.0)

# Import files for face recognition

print os.getcwd()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    # Capture frame-by-frame
    frame = vs.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_FEATURE_PARAMS_HAAR
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
vs.release()
cv2.destroyAllWindows()