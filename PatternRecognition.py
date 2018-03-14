import cv2
import copy
import numpy as np

# Uncomment this line to use the internal camera
from scipy.integrate._ivp.common import num_jac

cam = cv2.VideoCapture(0)



def recogniseTarget():
    pass

while True:
    # If local camera uncomment this
    ret, frame = cam.read()
    # frame = cam.read()

    template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('Frame', frame)

    # Change the picture to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Gray', gray)

    width, height = gray.shape
    normalized = np.zeros((width, height))

    normalized = cv2.normalize(gray, normalized, 50, 255, cv2.NORM_MINMAX)

    cv2.imshow('Normalized', normalized)

    # Calculate the binary threshold -> Everything lower than threshold turns black, everything over turns white
    ret_thresh, threshold = cv2.threshold(normalized, 100, 255, cv2.THRESH_BINARY)

    # Show the calculated threshold
    cv2.imshow('Threshold', threshold)

    MAX_MATCHES = 500
    MATCH_PERCENT = 0.01

    orb = cv2.ORB_create(MAX_MATCHES)

    template_keypoints, template_descriptors = orb.detectAndCompute(template, None)
    image_keypoints, image_descriptors = orb.detectAndCompute(threshold, None)

    image_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    feature_matches = image_matcher.match(template_descriptors, image_descriptors, None)

    feature_matches.sort(key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(feature_matches) * MATCH_PERCENT)

    feature_matches = feature_matches[:num_good_matches]

    matches_frame = cv2.drawMatches(template, template_keypoints, threshold, image_keypoints, feature_matches, None)

    match_points = np.zeros((len(feature_matches), 2), dtype=np.float32)

    for i, match in enumerate(feature_matches):
        match_points[i, :] = image_keypoints[match.trainIdx].pt

    matches_threshold_frame = cv2.cvtColor(threshold, cv2.COLOR_BAYER_GR2RGB)

    for point in match_points:
        point_coordinates = (point[0], point[1])
        cv2.drawMarker(matches_threshold_frame, point_coordinates, (255,0,0), 1, 6, 3)

    cv2.imshow('MarkThresh', matches_threshold_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Show the recognised rectangles
cam.release()
cv2.destroyAllWindows()