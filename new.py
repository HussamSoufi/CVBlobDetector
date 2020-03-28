import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 60, nothing)
cv2.createTrackbar("LS", "Tracking", 230, 230, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    frame = cv2.imread('4.jpeg')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    #cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Read image
im = cv2.imread("mask", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (1440, 880))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1 #10
params.maxThreshold = 2000 #200

# Filter by Area.
params.filterByArea = True # True
params.minArea = 50 #1500 

# Filter by Circularity
params.filterByCircularity = True #True
params.minCircularity = 0.1 #0.1

# Filter by Convexity
params.filterByConvexity = True #True
params.minConvexity = 0.0 #0.87

# Filter by Inertia
params.filterByInertia = True #True
params.minInertiaRatio = 0.0 #0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
total_count = 0
for i in keypoints:
    total_count = total_count + 1


im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

print(total_count)
