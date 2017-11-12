# Prepare the Raspberry Pi Camera
# https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
class VideoStream:
    def __init__(self, src=0, resolution=(640, 480),
                 framerate=24):
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