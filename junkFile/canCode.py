"""
Verision 1.0.1
Siddharth Gupta
------------------
   
1. Import necessary funcitons
2. Load image
3. Proccess image (grayscale, blur, etc.)
4. Create list of all detected circles between a certain radius
5. Unpack list of circles into x, y, r
6. Draw circles that are detected on the users screen
7. Show window
8. Wait for key press by user
9. If key is pressed, close windows 
"""

import cv2
import numpy as np

# load image
img = cv2.imread("/Users/pl1017970/Documents/circleDetection/betterCan.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur                
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
