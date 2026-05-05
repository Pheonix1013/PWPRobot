from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import cv2
import numpy as np
import os
import threading

app = FastAPI()

# --- GLOBALS ---
latest_frame = None
latest_frame_raw = None
autonomous_enabled = False
auto_command = "stop"
auto_error = 0
lane_status = "No lanes"
last_steering_cmd = "stop"

# Thread lock for globals written from executor + read from streaming endpoints
_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}

# Thresholds
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([140, 255, 255])
HORIZONTAL_MIN_WIDTH_RATIO = 0.45
HORIZONTAL_MAX_HEIGHT = 90
BOTTOM_STOP_ZONE_RATIO = 0.82

LANE_WIDTH_ESTIMATE = 260
CENTER_DEADBAND = 35
HARD_TURN_THRESHOLD = 85

# --- PERF: kernels created once at module level, not per-frame ---
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_BLUE_KERNEL = np.ones((5, 5), np.uint8)
_LINE_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
_LINE_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


@app.post("/stop")
async def stop():
    global autonomous_enabled, auto_command
    autonomous_enabled = False
    auto_command = "stop"
    for k in controls:
        controls[k] = False
    return {"message": "All movements stopped", "autonomous": autonomous_enabled}

@app.post("/autonomous/start")
async def autonomous_start():
    global autonomous_enabled, auto_command
    for k in controls:
        controls[k] = False
    autonomous_enabled = True
    auto_command = "stop"
    return {"autonomous": True}

@app.post("/autonomous/stop")
async def autonomous_stop():
    global autonomous_enabled, auto_command
    autonomous_enabled = False
    auto_command = "stop"
    return {"autonomous": False}


def get_mask(frame, hsv):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    med = int(np.median(blur))
    low = max(35, int(0.66 * med))
    high = min(220, int(1.33 * med))
    edges = cv2.Canny(blur, low, high)

    h, w = gray.shape
    roi_mask = np.zeros_like(edges)
    top_y = int(h * 0.45)
    poly = np.array([
        [int(w * 0.02), h - 1],
        [int(w * 0.98), h - 1],
        [int(w * 0.70), top_y],
        [int(w * 0.30), top_y],
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [poly], 255)

    mask = cv2.bitwise_and(edges, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _LINE_CLOSE_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _LINE_OPEN_KERNEL, iterations=1)
    mask = cv2.dilate(mask, _LINE_OPEN_KERNEL, iterations=1)
    return mask


def detect_blue_stop_line(frame, hsv):
    # PERF: hsv passed in — no redundant cvtColor call here
    h, w = frame.shape[:2]
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    # PERF: reuse module-level kernel
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, _BLUE_KERNEL)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, _BLUE_KERNEL)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_detected = False
    best_line = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 700:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        width_ratio = bw / float(w)
        bottom_y = y + bh
        if bw > bh * 2.5 and width_ratio >= HORIZONTAL_MIN_WIDTH_RATIO and bh <= HORIZONTAL_MAX_HEIGHT:
            if area > best_area:
                best_area = area
                line_y = y + bh // 2
                best_line = (x, line_y, x + bw, line_y, bottom_y >= int(h * BOTTOM_STOP_ZONE_RATIO))
    if best_line:
        x1, y1, x2, y2, is_near_bottom = best_line
        stop_detected = is_near_bottom
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255) if stop_detected else (255, 0, 0), 4)
    return stop_detected, frame


def average_line(lines):
    if not lines:
        return None

    arr = np.array(lines, dtype=np.float32)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    dx = x2 - x1
    valid = np.abs(dx) > 1e-3
    if np.count_nonzero(valid) < 2:
        return None

    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    lengths = np.hypot(x2 - x1, y2 - y1)
    slopes = (y2 - y1) / (x2 - x1)

    med_slope = np.median(slopes)
    mad = np.median(np.abs(slopes - med_slope)) + 1e-6
    inliers = np.abs(slopes - med_slope) <= (2.5 * mad + 0.12)

    if np.count_nonzero(inliers) < 2:
        inliers = np.ones_like(slopes, dtype=bool)

    pts_x = np.concatenate([x1[inliers], x2[inliers]])
    pts_y = np.concatenate([y1[inliers], y2[inliers]])
    w = np.concatenate([lengths[inliers], lengths[inliers]])

    if len(pts_x) < 2:
        return None

    fit = np.polyfit(pts_y, pts_x, 1, w=w)
    return (float(fit[0]), float(fit[1]))


def line_points_from_fit(fit, y1, y2, w):
    if fit is None:
        return None
    m, b = fit
    x1, x2 = int(m * y1 + b), int(m * y2 + b)
    return (max(0, min(w - 1, x1)), int(y1), max(0, min(w - 1, x2)), int(y2))


def draw_line(img, fit, color, thickness=4):
    h, w = img.shape[:2]
    p = line_points_from_fit(fit, h - 1, 0, w)
    if p:
        cv2.line(img, (p[0], p[1]), (p[2], p[3]), color, thickness)


def classify_lines(lines, w, h):
    left_lines, right_lines = [], []
    if lines is None:
        return left_lines, right_lines

    y_look = int(h * 0.82)
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            continue

        seg_len = float(np.hypot(dx, dy))
        if seg_len < 35:
            continue

        slope = dy / dx
        abs_slope = abs(slope)
        if abs_slope < 0.35 or abs_slope > 4.2:
            continue

        b = y1 - slope * x1
        x_at_look = (y_look - b) / slope
        x_bottom = ((h - 1) - b) / slope

        if not (-0.25 * w <= x_at_look <= 1.25 * w):
            continue
        if not (-0.25 * w <= x_bottom <= 1.25 * w):
            continue

        if slope < 0:
            if x_at_look < w * 0.57 and x_bottom < w * 0.72:
                left_lines.append((x1, y1, x2, y2))
        else:
            if x_at_look > w * 0.43 and x_bottom > w * 0.28:
                right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def compute_auto_command(left_fit, right_fit, w, h):
    global lane_status, last_steering_cmd, auto_error
    y_look = int(h * 0.82)
    frame_center = w // 2
    x_left = int(left_fit[0] * y_look + left_fit[1]) if left_fit else None
    x_right = int(right_fit[0] * y_look + right_fit[1]) if right_fit else None

    if x_left is not None and x_right is not None:
        lane_center = (x_left + x_right) // 2
        lane_center = max(0, min(w - 1, lane_center))
        error = lane_center - frame_center
        auto_error = error
        lane_status = f"Both lanes | err={error}"

        if abs(error) <= CENTER_DEADBAND:
            if last_steering_cmd not in ("left", "right"):
                last_steering_cmd = "forward"
            return "forward", lane_center, error

        cmd = "left" if error < 0 else "right"
        last_steering_cmd = cmd
        return cmd, lane_center, error

    if x_left is not None:
        # Left boundary seen, recover by turning right to bring right boundary back.
        auto_error = CENTER_DEADBAND + 1
        lane_status = "Left lane only | recovering right"
        last_steering_cmd = "right"
        lane_center = max(0, min(w - 1, x_left + LANE_WIDTH_ESTIMATE // 2))
        return "right", lane_center, auto_error

    if x_right is not None:
        # Right boundary seen, recover by turning left to bring left boundary back.
        auto_error = -(CENTER_DEADBAND + 1)
        lane_status = "Right lane only | recovering left"
        last_steering_cmd = "left"
        lane_center = max(0, min(w - 1, x_right - LANE_WIDTH_ESTIMATE // 2))
        return "left", lane_center, auto_error

    if last_steering_cmd == "right":
        lane_status = "No lanes | searching right"
        auto_error = 80
        return "right", None, 80
    if last_steering_cmd == "left":
        lane_status = "No lanes | searching left"
        auto_error = -80
        return "left", None, -80

    lane_status = "No lanes | searching forward"
    auto_error = 0
    return "forward", None, 0


def process_frame(frame):
    global auto_command, autonomous_enabled, lane_status
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # PERF: single HSV conversion shared by both functions
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    stop_detected, frame = detect_blue_stop_line(frame, hsv)
    mask = get_mask(frame, hsv)

    roi_y = int(h * 0.45)
    roi = mask[roi_y:, :]
    lines_raw = cv2.HoughLinesP(roi, 1, np.pi / 180, 28, minLineLength=45, maxLineGap=65)

    lines = None
    if lines_raw is not None:
        lines = lines_raw.copy()
        lines[:, :, 1] += roi_y
        lines[:, :, 3] += roi_y

    l_lines, r_lines = classify_lines(lines, w, h)
    l_fit, r_fit = average_line(l_lines), average_line(r_lines)

    draw_line(frame, l_fit, (0, 0, 255))
    draw_line(frame, r_fit, (255, 0, 0))
    cmd, l_center, err = compute_auto_command(l_fit, r_fit, w, h)

    f_center, y_look = w // 2, int(h * 0.82)
    cv2.line(frame, (f_center, h - 1), (f_center, 0), (255, 255, 0), 2)
    cv2.line(frame, (0, y_look), (w, y_look), (255, 255, 0), 2)
    cv2.line(frame, (0, roi_y), (w, roi_y), (80, 180, 255), 1)
    if l_center is not None:
        cv2.circle(frame, (l_center, y_look), 8, (0, 255, 0), -1)
    cv2.putText(frame, f"err={err}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if stop_detected:
        auto_command = "stop"
        if autonomous_enabled:
            autonomous_enabled = False
            lane_status = "Stop Line Reached"
    else:
        auto_command = cmd
    return frame


@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, latest_frame_raw
    data = await request.json()
    frame_b64 = data.get("frame")
    if not frame_b64:
        return {"status": "no frame"}
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "bad frame"}
    loop = asyncio.get_running_loop()
    processed = await loop.run_in_executor(None, process_frame, frame)
    ok, jpeg = cv2.imencode(".jpg", processed)
    if not ok:
        return {"status": "encode failed"}
    # PERF: lock guards globals written here and read by streaming endpoints
    with _lock:
        latest_frame = jpeg.tobytes()
        latest_frame_raw = frame_bytes
    return {"status": "ok", "auto_command": auto_command}


@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            with _lock:
                frame = latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_raw")
async def video_feed_raw():
    async def stream():
        while True:
            with _lock:
                frame = latest_frame_raw
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui.html")
    try:
        with open(html_path) as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>gui.html not found</h1>", status_code=404)


@app.get("/status")
async def status():
    m_cmd = "stop"
    for k in controls:
        if controls[k]:
            m_cmd = k
            break
    return {
        "controls": controls,
        "manual_command": m_cmd,
        "autonomous": autonomous_enabled,
        "auto_command": auto_command,
        "auto_error": auto_error,
        "lane_status": lane_status,
    }


@app.post("/{direction}")
async def move(direction: str):
    global autonomous_enabled, auto_command
    if direction not in controls:
        return {"error": "invalid direction"}
    autonomous_enabled = False
    auto_command = "stop"
    for k in controls:
        controls[k] = False
    controls[direction] = True
    return {direction: True, "autonomous": autonomous_enabled}
