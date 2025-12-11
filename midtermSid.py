import cv2
import numpy as np
from collections import deque

# --- Parameters ---
frame_width = 640
frame_height = 480
blur_ksize = 15
canny_low = 50
canny_high = 150
hough_threshold = 50
min_line_length = 80
max_line_gap = 30

# how many sample points vertically to take
num_samples = 20

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

def line_at_y(x1,y1,x2,y2, y_target):
    """Return X position on the line at a specific Y."""
    if y2 == y1:
        return None
    t = (y_target - y1) / (y2 - y1)
    if 0 <= t <= 1:
        x = x1 + t * (x2 - x1)
        return int(x)
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    left_line = None
    right_line = None

    # find the two longest-ish lines and treat them as left/right
    if lines is not None:
        # sort by length
        lines_sorted = sorted(lines, key=lambda L: (L[0][2]-L[0][0])**2 + (L[0][3]-L[0][1])**2, reverse=True)
        if len(lines_sorted) >= 2:
            left_line = lines_sorted[0][0]
            right_line = lines_sorted[1][0]

    if left_line is not None and right_line is not None:
        x1l,y1l,x2l,y2l = left_line
        x1r,y1r,x2r,y2r = right_line

        # draw left + right lines
        cv2.line(frame, (x1l,y1l),(x2l,y2l),(255,0,0),2)
        cv2.line(frame, (x1r,y1r),(x2r,y2r),(255,0,0),2)

        # sample midpoints
        midpoints = []
        for i in range(num_samples):
            y = int(frame_height * i / (num_samples - 1))

            lx = line_at_y(x1l,y1l,x2l,y2l, y)
            rx = line_at_y(x1r,y1r,x2r,y2r, y)

            if lx is not None and rx is not None:
                mid_x = (lx + rx) // 2
                midpoints.append((mid_x, y))

        # draw the center line from all midpoints
        if len(midpoints) > 1:
            for i in range(len(midpoints)-1):
                cv2.line(frame, midpoints[i], midpoints[i+1], (0,0,255), 2)

    cv2.imshow("Centerline From Midpoints", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
