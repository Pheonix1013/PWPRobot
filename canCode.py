import cv2
import numpy as np

# load image
img = cv2.imread("/Users/pl1017970/Documents/circleDetection/betterCan.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur to reduce noise
gray = cv2.medianBlur(gray, 11)

# detect circles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=500,
    param1=120,
    param2=45,
    minRadius=500,
    maxRadius=1000
)

# draw circles
"""
if circles is not None:
    circles = circles[0]
    x, y, r = max(circles, key=lambda c: c[2])
    cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 4)
    cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)

"""
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        #distance = np.sqrt((x-  WORK ON THIS, GET CLOSEST DISTANCE FROM CENTER AND USE THAT POINT TO DO WTVR
        if x==x:
            cv2.circle(img, (x, y), r, (0, 255, 0), 10)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
            print(x,y,r)

        #cv2.circle(img, (x, y), r, (0, 255, 0), 4)
# show result
cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
