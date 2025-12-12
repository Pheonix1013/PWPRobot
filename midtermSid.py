import cv2
import numpy as np

FRAME_W = 640
FRAME_H = 480

#stoering the variables (so i can fine tune them for testing)
BLUR_K = 9
TH_BLOCK = 51
TH_C = 7
MORPH_K = 7

#more variables - for contour
MIN_ARCLEN = 150.0
MIN_AREA = 200

#number of samples that we gonna take for the line of best fit
NUM_SAMPLES = 60
SMOOTH_WIN = 7

# drawing
CROP_X, CROP_Y, CROP_W, CROP_H = 160, 120, 320, 240
LINE_THICK = 3


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

def pleaseWork(contour_pts):
    pts = contour_pts.reshape(-1, 2).astype(float)
    xs = pts[:, 0]
    ys = pts[:, 1]
    best = ("v", 0, 0, 1e12)
    try:
        m_v, b_v = np.polyfit(ys, xs, 1)
        err_v = np.mean((xs - (m_v*ys + b_v))**2)
        best = ("v", m_v, b_v, err_v)
    except Exception:
        pass
    try:
        m_h, b_h = np.polyfit(xs, ys, 1)
        err_h = np.mean((ys - (m_h*xs + b_h))**2)
        if err_h < best[3]:
            best = ("h", m_h, b_h, err_h)
    except Exception:
        pass
    return best

def checkIfMoving(a, n):
    if len(a) < n:
        return a
    return np.convolve(a, np.ones(n)/n, mode='same')

def extendLines(mode, m, b, width, height):
    if mode == "v":
        x0 = m*0 + b
        x1 = m*(height-1) + b
        p1 = (int(round(x0)), 0)
        p2 = (int(round(x1)), height-1)
    else:
        y0 = m*0 + b
        y1 = m*(width-1) + b
        p1 = (0, int(round(y0)))
        p2 = (width-1, int(round(y1)))
    return p1, p2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x2 = min(CROP_X + CROP_W, FRAME_W)
    y2 = min(CROP_Y + CROP_H, FRAME_H)
    cropped = frame[CROP_Y:y2, CROP_X:x2].copy()
    h = cropped.shape[0]
    w = cropped.shape[1]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, TH_BLOCK, TH_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for cnt in contours:
        al = cv2.arcLength(cnt, closed=False)
        area = cv2.contourArea(cnt)
        if al >= MIN_ARCLEN and area >= MIN_AREA:
            good.append((al, cnt))

    if len(good) < 2:
        contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        for cnt in contours_sorted[:3]:
            if cv2.contourArea(cnt) >= 50:
                good.append((cv2.arcLength(cnt,False), cnt))

    good = sorted(good, key=lambda x: x[0], reverse=True)[:2]

    #list so that we can get the lines drawn so we can also draw them in main
    fullframe_lines = []

    if len(good) >= 2:
        cntA = good[0][1]
        cntB = good[1][1]

        def mean_x(cnt):
            pts = cnt.reshape(-1,2)
            return pts[:,0].mean()

        if mean_x(cntA) <= mean_x(cntB):
            left_cnt, right_cnt = cntA, cntB
        else:
            left_cnt, right_cnt = cntB, cntA

        modeL, mL, bL, _ = pleaseWork(left_cnt)
        modeR, mR, bR, _ = pleaseWork(right_cnt)

        # draw the contours in the cropped frame only not main frame yet
        cv2.drawContours(cropped, [left_cnt], -1, (0,0,255), 2)
        cv2.drawContours(cropped, [right_cnt], -1, (255,0,0), 2)

        #get midpoints to draw the best fit line using averageing of the slope

        midpoints = []

        if modeL == "v" and modeR == "v":
            ys = np.linspace(0, h-1, NUM_SAMPLES)
            for yy in ys:
                xL = mL*yy + bL
                xR = mR*yy + bR
                midpoints.append(((xL+xR)/2.0, yy))
        elif modeL == "h" and modeR == "h":
            xs = np.linspace(0, w-1, NUM_SAMPLES)
            for xx in xs:
                yL = mL*xx + bL
                yR = mR*xx + bR
                midpoints.append((xx, (yL+yR)/2.0))
        else:
            ts = np.linspace(0.0, 1.0, NUM_SAMPLES)
            for t in ts:
                xx = t*(w-1)
                yy = t*(h-1)
                if modeL == "v":
                    xL = mL*yy + bL; yL = yy
                else:
                    yL = mL*xx + bL; xL = xx
                if modeR == "v":
                    xR = mR*yy + bR; yR = yy
                else:
                    yR = mR*xx + bR; xR = xx
                midpoints.append(((xL+xR)/2.0, (yL+yR)/2.0))

        if len(midpoints) >= 3:
            pts = np.array(midpoints)
            xs = checkIfMoving(pts[:,0], SMOOTH_WIN)
            ys = checkIfMoving(pts[:,1], SMOOTH_WIN)
            mid_s = np.vstack((xs, ys)).T

            #bunch of try and excepts so that code dont break and show errors
            #before what happened was when lines werent lined up or cropped box wasnt in the right spot it would break because it was trying to get the midpoint of 0, so undefined so error
            #now it gets midpoints of edge of the cropped box
            err_v = err_h = 1e12
            try:
                mcv, bcv = np.polyfit(mid_s[:,1], mid_s[:,0], 1)
                err_v = np.mean((mid_s[:,0] - (mcv*mid_s[:,1] + bcv))**2)
            except: pass
            try:
                mch, bch = np.polyfit(mid_s[:,0], mid_s[:,1], 1)
                err_h = np.mean((mid_s[:,1] - (mch*mid_s[:,0] + bch))**2)
            except: pass

            if err_v <= err_h:
                p_top = (int(round(mcv*0 + bcv)), 0)
                p_bot = (int(round(mcv*(h-1) + bcv)), h-1)
            else:
                p_top = (0, int(round(mch*0 + bch)))
                p_bot = (w-1, int(round(mch*(w-1) + bch)))

            def clip(p):
                return (int(max(0,min(w-1,p[0]))), int(max(0,min(h-1,p[1]))))

            p1 = clip(p_top)
            p2 = clip(p_bot)

            # draw center line in cropped
            cv2.line(cropped, p1, p2, (0,255,0), LINE_THICK)

            #just add the cropped frame x and y to the lines so that the lines don't show in the wrong place for the main frame
            p1_abs = (p1[0] + CROP_X, p1[1] + CROP_Y)
            p2_abs = (p2[0] + CROP_X, p2[1] + CROP_Y)
            fullframe_lines.append((p1_abs, p2_abs, (0,255,0)))

        #draw left (red) line
        pL1, pL2 = extendLines(modeL, mL, bL, w, h)
        cv2.line(cropped, pL1, pL2, (0,0,255), LINE_THICK)
        fullframe_lines.append(((pL1[0]+CROP_X, pL1[1]+CROP_Y),
                                (pL2[0]+CROP_X, pL2[1]+CROP_Y),
                                (0,0,255)))

        #draw right (blue) line
        pR1, pR2 = extendLines(modeR, mR, bR, w, h)
        cv2.line(cropped, pR1, pR2, (255,0,0), LINE_THICK)
        fullframe_lines.append(((pR1[0]+CROP_X, pR1[1]+CROP_Y),
                                (pR2[0]+CROP_X, pR2[1]+CROP_Y),
                                (255,0,0)))

    # draw crop rectangle box
    cv2.rectangle(frame, (CROP_X, CROP_Y), (x2, y2), (0,0,255), 2)

    #draw all lines on the main frame instead of just crop
    for (pa, pb, col) in fullframe_lines:
        cv2.line(frame, pa, pb, col, LINE_THICK)

    #ahow full frame
    cv2.imshow("Full frame", frame)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
