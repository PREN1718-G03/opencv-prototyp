import cv2
import copy
import time
from VideoStream import VideoStream

# Comment these two lines if you want to use the internal camera
cam  = VideoStream().start()
# Let the camera sensor warm up
time.sleep(2.0)

# Uncomment this line to use the internal camera
# cam = cv2.VideoCapture(1)



def recogniseTarget(coordinateArray):
    # How near the two middle points should be in pixels
    ERRMAX = 4
    match = 0
    found = False
    targetCoordinates = (None, None)
    comparedCoordinates = (0, 0)
    for coordinates in coordinateArray:
        differenceX = abs(comparedCoordinates[0]-coordinates[0])
        differenceY = abs(comparedCoordinates[1]-coordinates[1])
        if differenceX < ERRMAX and differenceY < ERRMAX:
            match +=1
        else:
            match = 0
        comparedCoordinates = coordinates
        if match > 2:
            found = True
            targetCoordinates = coordinates
    return targetCoordinates, found


while True:
    # If local camera uncomment this
    # ret, frame = cam.read()
    frame = cam.read()

    # Change the picture to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the binary threshold -> Everything lower than threshold turns black, everything over turns white
    retthresh, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Show the calculated threshold
    cv2.imshow('Threshold', threshold)

    # Find the contours, RETR_TREE preserves the inner contours and doesn't limit itself to the outermost one like RETR_EXTERNAL
    imageContours, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # I don't want a reference to the frame object, I want a copy
    cntImage = copy.copy(frame)

    crnImage = copy.copy(frame)

    cv2.drawContours(cntImage, contours, -1, (0, 255, 0), 2)
    # Show the picture of the contours
    cv2.imshow('Contours', cntImage)

    # I want to enumerate the found rectangles
    rectCount = 0
    coordinateArray = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # Epsilon = Maximum Distance from contour to approximated contour; Usually 1-5% of ArcLength,
        # we're setting it to 15% :D
        epsilon = 0.15 * perimeter
        approximatedContour = cv2.approxPolyDP(contour, epsilon, True)

        contourArea = cv2.contourArea(approximatedContour)
        if len(approximatedContour) == 4 and contourArea > 50.0:
            rectCount += 1

            # Calculate the corner points of the rectangle
            cornerPoints = cv2.boxPoints(cv2.minAreaRect(approximatedContour))

            # Calculate the centroid from moments https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            moments = cv2.moments(approximatedContour)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            coordinates = (cx, cy)
            coordinateArray.append(coordinates)

            cv2.drawMarker(frame, coordinates, (170 + rectCount * 20, 0, 170 + rectCount * 20), 1, 6, 3)
            cv2.drawContours(frame, [approximatedContour], -1, (0, 255, 255), 1)

            cv2.putText(frame, str(rectCount), tuple(cornerPoints[0]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 200))

        print coordinateArray
        t,f = recogniseTarget(coordinateArray)
        print f

    # corners = cv2.cornerHarris(crnImage,)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Show the recognised rectangles
    cv2.imshow('Rectangles', frame)
cam.release()
cv2.destroyAllWindows()