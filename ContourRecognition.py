import cv2
import copy
cam = cv2.VideoCapture(1)
while True:
    ret, frame = cam.read()

    # Change the picture to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the binary threshold -> Everything lower than threshold turns black, everything over turns white
    retthresh, threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', threshold)

    # Find the contours, RETR_TREE preserves the inner contours and doesn't limit itself to the outermost one like RETR_EXTERNAL
    imageContours, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # I don't want a reference to the frame object, I want a copy
    cntImage = copy.copy(frame)
    cv2.drawContours(cntImage, contours, -1, (0,255,0), 2)
    cv2.imshow('Contours', cntImage)

    # I want to enumerate the found rectangles
    rectCount = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # Epsilon = Maximum Distance from contour to approximated contour; Usually 1-5% of ArcLength
        epsilon = 0.15*perimeter
        approximatedContour = cv2.approxPolyDP(contour, epsilon, True)

        contourArea = cv2.contourArea(approximatedContour)
        if len(approximatedContour) == 4 and contourArea>20.0:
            rectCount += 1
            # Calculate the corner points of the rectangle
            cornerPoints = cv2.boxPoints(cv2.minAreaRect(approximatedContour))
            # Calculate the centroid from moments https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
            moments = cv2.moments(approximatedContour)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            coordinates = (cx,cy)
            print coordinates
            cv2.drawContours(frame, [approximatedContour], -1, (0, 0, 255), 3)

            cv2.putText(frame, str(rectCount), tuple(cornerPoints[0]), cv2.FONT_HERSHEY_PLAIN, 2 , (0,0,200))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Rectangles', frame)
cam.release()
cv2.destroyAllWindows()