import cv2
import numpy as np

# params
frame_width = 640
frame_height = 480
blur_ksize = 9
thresh_blocksize = 51
thresh_C = 7
morph_kernel = 7
min_line_area = 300  # tune: min contour area to consider a stroke
num_samples = 30     # how many horizontal slices to sample for midpoints
smooth_window = 5    # smooth center x over neighbors

# crop (same as yours)
x, y, w, h = 160, 120, 320, 240

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


def sample_x_from_mask(mask, yy):
    """Return the center x of mask at row yy (relative to mask coords) or None."""
    # get all x positions where mask is nonzero at row yy
    row = mask[yy, :]
    xs = np.where(row > 0)[0]
    if xs.size == 0:
        return None
    # choose center of the stroke at that row
    return int((int(xs[0]) + int(xs[-1])) / 2)


def moving_average(a, n=3):
    if len(a) < n:
        return a
    ret = np.convolve(a, np.ones(n)/n, mode='same')
    return ret.astype(int)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    x2 = min(x + w, frame_width)
    y2 = min(y + h, frame_height)
    cropped = frame[y:y2, x:x2].copy()

    # 1) preprocess -> create a binary mask of the dark strokes (works well for black lines on white)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # adaptive threshold (better for uneven lighting), invert so strokes are white
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_C)

    # morphological close to fill stroke interior and remove small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2) find contours -> pick the two largest as the two side strokes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    strokes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_line_area:
            strokes.append(cnt)
        if len(strokes) >= 2:
            break

    # if we have at least two stroke-contours, make separate masks for them
    center_pts = []
    if len(strokes) >= 2:
        # create blank masks for left/right stroke
        stroke_mask1 = np.zeros_like(mask)
        stroke_mask2 = np.zeros_like(mask)
        cv2.drawContours(stroke_mask1, [strokes[0]], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(stroke_mask2, [strokes[1]], -1, 255, thickness=cv2.FILLED)

        # optional: ensure stroke1 is left of stroke2 by comparing average x
        def avg_x_of_mask(st_mask):
            cols = np.where(st_mask.sum(axis=0) > 0)[0]
            return cols.mean() if cols.size > 0 else 1e9

        if avg_x_of_mask(stroke_mask1) > avg_x_of_mask(stroke_mask2):
            # swap so stroke_mask1 is left, stroke_mask2 is right
            stroke_mask1, stroke_mask2 = stroke_mask2, stroke_mask1

        # sample many horizontal slices across the crop
        midpoints = []
        for i in range(num_samples):
            yy = int((h - 1) * i / (num_samples - 1))  # row in [0, h-1]

            lx = sample_x_from_mask(stroke_mask1, yy)
            rx = sample_x_from_mask(stroke_mask2, yy)

            if lx is not None and rx is not None:
                # midpoint in cropped coordinates
                mid_x = (lx + rx) // 2
                midpoints.append((mid_x, yy))
                center_pts.append(mid_x)

        # smooth center x positions (moving average) to reduce jitter
        if center_pts:
            smoothed = moving_average(np.array(center_pts), n=smooth_window)
            # replace midpoints x with smoothed values
            for idx in range(min(len(midpoints), len(smoothed))):
                midpoints[idx] = (int(smoothed[idx]), midpoints[idx][1])

        # draw left/right stroke bounding (for debug) and centerline
        cv2.drawContours(cropped, [strokes[0]], -1, (0,0,255), 2)  # left = red
        cv2.drawContours(cropped, [strokes[1]], -1, (255,0,0), 2)  # right = blue

        if len(midpoints) > 1:
            for i in range(len(midpoints)-1):
                cv2.line(cropped, midpoints[i], midpoints[i+1], (0,255,0), 3)  # center = green

    # preview crop box on full frame
    cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)

    cv2.imshow("Full frame", frame)
    cv2.imshow("Cropped", cropped)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
