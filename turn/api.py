from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import cv2
import numpy as np
import os

app = FastAPI()

# --- GLOBALS ---
latest_frame = None
latest_frame_raw = None
autonomous_enabled = False
auto_command = "stop"
auto_error = 0
lane_status = "No lanes"

# Memory for recovery: tracks the last known lane configuration
# Options: "both", "left", "right"
last_seen_lane = "both"

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
CENTER_DEADBAND = 35  # Higher deadband helps re-straighten without jitter
HARD_TURN_THRESHOLD = 85

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

def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    s_blur = cv2.GaussianBlur(s, (7, 7), 0)
    v_blur = cv2.GaussianBlur(v, (7, 7), 0)
    s_bin = cv2.adaptiveThreshold(s_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    v_bin = cv2.adaptiveThreshold(v_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    mask = cv2.bitwise_and(s_bin, v_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def detect_blue_stop_line(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_detected = False
    best_line = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 700: continue
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
    if not lines: return None
    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2]); ys.extend([y1, y2])
    if len(xs) < 2: return None
    fit = np.polyfit(np.array(ys, dtype=np.float32), np.array(xs, dtype=np.float32), 1)
    return (fit[0], fit[1])

def line_points_from_fit(fit, y1, y2, w):
    if fit is None: return None
    m, b = fit
    x1, x2 = int(m * y1 + b), int(m * y2 + b)
    return (max(0, min(w - 1, x1)), int(y1), max(0, min(w - 1, x2)), int(y2))

def draw_line(img, fit, color, thickness=4):
    h, w = img.shape[:2]
    p = line_points_from_fit(fit, h - 1, 0, w)
    if p: cv2.line(img, (p[0], p[1]), (p[2], p[3]), color, thickness)

def classify_lines(lines, w, h):
    left_lines, right_lines = [], []
    if lines is None: return left_lines, right_lines
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        if abs(slope) < 0.45 or ((x2 - x1)**2 + (y2 - y1)**2)**0.5 < 30: continue
        if slope < 0 and mid_x < w * 0.72: left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > w * 0.28: right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines

def compute_auto_command(left_fit, right_fit, w, h):
    global lane_status, last_seen_lane, auto_error
    y_look = int(h * 0.82)
    frame_center = w // 2
    x_left = int(left_fit[0] * y_look + left_fit[1]) if left_fit else None
    x_right = int(right_fit[0] * y_look + right_fit[1]) if right_fit else None

    if x_left is not None and x_right is not None:
        last_seen_lane = "both"
        lane_center = (x_left + x_right) // 2
        mode = "Both lanes"
    elif x_left is not None:
        last_seen_lane = "left"
        lane_center = x_left + LANE_WIDTH_ESTIMATE // 2
        mode = "Left lane only (Turning Right)"
    elif x_right is not None:
        last_seen_lane = "right"
        lane_center = x_right - LANE_WIDTH_ESTIMATE // 2
        mode = "Right lane only (Turning Left)"
    else:
        # Recovery: turn toward where we last saw a lane boundary
        if last_seen_lane == "left":
            lane_status = "Lanes Lost: Searching Right"
            auto_error = 100
            return "right", None, 100
        elif last_seen_lane == "right":
            lane_status = "Lanes Lost: Searching Left"
            auto_error = -100
            return "left", None, -100
        else:
            lane_status = "No lanes/memory"
            auto_error = 0
            return "stop", None, 0

    lane_center = max(0, min(w - 1, lane_center))
    error = lane_center - frame_center
    auto_error = error
    lane_status = f"{mode} | err={error}"

    if abs(error) <= CENTER_DEADBAND: return "forward", lane_center, error
    return ("left" if error < 0 else "right"), lane_center, error

def process_frame(frame):
    global auto_command, autonomous_enabled, lane_status
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    stop_detected, frame = detect_blue_stop_line(frame)
    mask = get_mask(frame)
    mask[:int(h * 0.45), :] = 0  # ignore top 45% — ceiling/walls, not track
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 35, minLineLength=30, maxLineGap=30)
    l_lines, r_lines = classify_lines(lines, w, h)
    l_fit, r_fit = average_line(l_lines), average_line(r_lines)
    
    draw_line(frame, l_fit, (0, 0, 255)); draw_line(frame, r_fit, (255, 0, 0))
    cmd, l_center, err = compute_auto_command(l_fit, r_fit, w, h)

    f_center, y_look = w // 2, int(h * 0.82)
    cv2.line(frame, (f_center, h-1), (f_center, 0), (255, 255, 0), 2)
    cv2.line(frame, (0, y_look), (w, y_look), (255, 255, 0), 2)
    if l_center is not None:
        cv2.circle(frame, (l_center, y_look), 8, (0, 255, 0), -1)

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
    if not frame_b64: return {"status": "no frame"}
    frame_bytes = base64.b64decode(frame_b64)
    latest_frame_raw = frame_bytes
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None: return {"status": "bad frame"}
    loop = asyncio.get_running_loop()
    processed = await loop.run_in_executor(None, process_frame, frame)
    ok, jpeg = cv2.imencode(".jpg", processed)
    if not ok: return {"status": "encode failed"}
    latest_frame = jpeg.tobytes()
    return {"status": "ok", "auto_command": auto_command}

@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            if latest_frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_raw")
async def video_feed_raw():
    async def stream():
        while True:
            if latest_frame_raw:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame_raw + b"\r\n")
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
        if controls[k]: m_cmd = k; break
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
    if direction not in controls: return {"error": "invalid direction"}
    autonomous_enabled = False
    auto_command = "stop"
    for k in controls: controls[k] = False
    controls[direction] = True
    return {direction: True, "autonomous": autonomous_enabled}
