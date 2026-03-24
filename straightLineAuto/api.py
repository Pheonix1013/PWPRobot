from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import cv2
import numpy as np

app = FastAPI()

latest_frame = None
latest_frame_raw = None

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

autonomous_enabled = False
auto_command = "stop"
lane_status = "No lanes"

BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([140, 255, 255])

HORIZONTAL_MIN_WIDTH_RATIO = 0.45
HORIZONTAL_MAX_HEIGHT = 90
BOTTOM_STOP_ZONE_RATIO = 0.82

LANE_WIDTH_ESTIMATE = 260
CENTER_DEADBAND = 35
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

    s_bin = cv2.adaptiveThreshold(
        s_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -10
    )

    v_bin = cv2.adaptiveThreshold(
        v_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, -10
    )

    mask = cv2.bitwise_and(s_bin, v_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    mask = cv2.bitwise_or(mask, edges)
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
        if area < 700:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        width_ratio = bw / float(w)
        bottom_y = y + bh

        is_horizontal = bw > bh * 2.5
        is_wide_enough = width_ratio >= HORIZONTAL_MIN_WIDTH_RATIO
        is_short_enough = bh <= HORIZONTAL_MAX_HEIGHT
        is_near_bottom = bottom_y >= int(h * BOTTOM_STOP_ZONE_RATIO)

        if is_horizontal and is_wide_enough and is_short_enough:
            if area > best_area:
                best_area = area
                line_y = y + bh // 2
                best_line = (x, line_y, x + bw, line_y, is_near_bottom)

    if best_line is not None:
        x1, y1, x2, y2, is_near_bottom = best_line
        stop_detected = is_near_bottom
        color = (0, 0, 255) if stop_detected else (255, 0, 0)
        cv2.line(frame, (x1, y1), (x2, y2), color, 4)

    return stop_detected, frame


def average_line(lines):
    if not lines:
        return None

    xs = []
    ys = []

    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if len(xs) < 2:
        return None

    fit = np.polyfit(np.array(ys, dtype=np.float32), np.array(xs, dtype=np.float32), 1)
    m, b = fit[0], fit[1]
    return (m, b)


def line_points_from_fit(fit, y1, y2, w):
    if fit is None:
        return None

    m, b = fit
    x1 = int(m * y1 + b)
    x2 = int(m * y2 + b)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))

    return (x1, int(y1), x2, int(y2))


def draw_line(img, fit, color, thickness=4):
    h, w = img.shape[:2]
    if fit is None:
        return

    p = line_points_from_fit(fit, h - 1, 0, w)
    if p is None:
        return

    x1, y1, x2, y2 = p
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def classify_lines(lines, w, h):
    left_lines = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    for l in lines:
        x1, y1, x2, y2 = l[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        mid_x = (x1 + x2) / 2

        if abs(slope) < 0.45:
            continue
        if length < 30:
            continue

        if slope < 0 and mid_x < w * 0.72:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > w * 0.28:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def compute_auto_command(left_fit, right_fit, w, h):
    global lane_status

    y_look = int(h * 0.82)
    frame_center = w // 2

    x_left = None
    x_right = None
    lane_center = None
    mode = None

    if left_fit is not None:
        x_left = int(left_fit[0] * y_look + left_fit[1])

    if right_fit is not None:
        x_right = int(right_fit[0] * y_look + right_fit[1])

    if x_left is not None and x_right is not None:
        lane_center = (x_left + x_right) // 2
        mode = "Both lanes"
    elif x_left is not None:
        lane_center = x_left + LANE_WIDTH_ESTIMATE // 2
        mode = "Left lane only"
    elif x_right is not None:
        lane_center = x_right - LANE_WIDTH_ESTIMATE // 2
        mode = "Right lane only"
    else:
        lane_status = "No lanes"
        return "stop", None, None

    lane_center = max(0, min(w - 1, lane_center))
    error = lane_center - frame_center

    lane_status = f"{mode} | err={error}"

    if abs(error) <= CENTER_DEADBAND:
        return "forward", lane_center, error

    if error < -HARD_TURN_THRESHOLD:
        return "left", lane_center, error

    if error > HARD_TURN_THRESHOLD:
        return "right", lane_center, error

    if error < 0:
        return "left", lane_center, error

    return "right", lane_center, error


def process_frame(frame):
    global auto_command, autonomous_enabled, lane_status

    frame = cv2.resize(frame, (640, 480))

    stop_detected, frame = detect_blue_stop_line(frame)

    mask = get_mask(frame)

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=30
    )

    h, w = frame.shape[:2]

    left_lines, right_lines = classify_lines(lines, w, h)

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    draw_line(frame, left_fit, (0, 0, 255), 4)
    draw_line(frame, right_fit, (255, 0, 0), 4)

    cmd, lane_center, error = compute_auto_command(left_fit, right_fit, w, h)

    frame_center = w // 2
    y_look = int(h * 0.82)

    cv2.line(frame, (frame_center, h - 1), (frame_center, 0), (255, 255, 0), 2)
    cv2.line(frame, (0, y_look), (w, y_look), (255, 255, 0), 2)

    if lane_center is not None:
        cv2.circle(frame, (lane_center, y_look), 8, (0, 255, 0), -1)
        cv2.line(frame, (frame_center, y_look), (lane_center, y_look), (0, 255, 255), 2)

    no_current_lines = left_fit is None and right_fit is None

    if stop_detected:
        auto_command = "stop"
        if autonomous_enabled:
            autonomous_enabled = False
            lane_status = "Blue stop line reached | automation stopped"
        else:
            lane_status = "Blue stop line detected"
    elif no_current_lines:
        auto_command = "stop"
        lane_status = "No lanes"
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
    latest_frame_raw = frame_bytes

    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "bad frame"}

    processed = process_frame(frame)

    ok, jpeg = cv2.imencode(".jpg", processed)
    if not ok:
        return {"status": "encode failed"}

    latest_frame = jpeg.tobytes()
    return {"status": "ok", "auto_command": auto_command}


@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            if latest_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    latest_frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_raw")
async def video_feed_raw():
    async def stream():
        while True:
            if latest_frame_raw:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    latest_frame_raw +
                    b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>PWP Robot Control</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #111; color: white; }
        table { width: 100vw; height: 100vh; border-collapse: collapse; }
        td { border: 1px solid #333; text-align: center; vertical-align: middle; }
        button {
          width: 110px;
          height: 55px;
          font-size: 18px;
          margin: 6px;
          cursor: pointer;
        }
        .play { background: #1f9d55; color: white; }
        .pause { background: #c53030; color: white; }
        .manual { background: #2b6cb0; color: white; }
        #statusBox { font-size: 20px; line-height: 1.8; }
      </style>
    </head>
    <body>
      <table>
        <tr height="50%">
          <td width="50%">
            <h3>Processed Feed</h3>
            <img src="/video_feed" width="480" height="320">
          </td>
          <td width="50%">
            <h3>Controls</h3>
            <table style="margin:auto; border-collapse:collapse;">
              <tr>
                <td></td>
                <td><button class="manual" onclick="sendCommand('forward')">&#8593;</button></td>
                <td></td>
              </tr>
              <tr>
                <td><button class="manual" onclick="sendCommand('left')">&#8592;</button></td>
                <td><button class="pause" onclick="stopMotor()">■</button></td>
                <td><button class="manual" onclick="sendCommand('right')">&#8594;</button></td>
              </tr>
              <tr>
                <td></td>
                <td><button class="manual" onclick="sendCommand('backward')">&#8595;</button></td>
                <td></td>
              </tr>
            </table>

            <div style="margin-top:20px;">
              <button class="play" onclick="startAuto()">▶ Start Auto</button>
              <button class="pause" onclick="stopAuto()">⏸ Stop Auto</button>
            </div>
          </td>
        </tr>

        <tr height="50%">
          <td width="50%">
            <h3>Raw Feed</h3>
            <img src="/video_feed_raw" width="480" height="320">
          </td>
          <td width="50%">
            <div id="statusBox">Loading status...</div>
          </td>
        </tr>
      </table>

      <script>
        const API_BASE = "";

        async function sendCommand(direction) {
          try {
            await fetch(`${API_BASE}/${direction}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" }
            });
          } catch (error) {
            console.error(error);
          }
        }

        async function stopMotor() {
          try {
            await fetch(`${API_BASE}/stop`, {
              method: "POST",
              headers: { "Content-Type": "application/json" }
            });
          } catch (error) {
            console.error(error);
          }
        }

        async function startAuto() {
          try {
            await fetch(`${API_BASE}/autonomous/start`, {
              method: "POST",
              headers: { "Content-Type": "application/json" }
            });
          } catch (error) {
            console.error(error);
          }
        }

        async function stopAuto() {
          try {
            await fetch(`${API_BASE}/autonomous/stop`, {
              method: "POST",
              headers: { "Content-Type": "application/json" }
            });
          } catch (error) {
            console.error(error);
          }
        }

        async function refreshStatus() {
          try {
            const response = await fetch(`${API_BASE}/status`);
            const data = await response.json();

            document.getElementById("statusBox").innerHTML = `
              <div><b>Autonomous:</b> ${data.autonomous}</div>
              <div><b>Manual command:</b> ${data.manual_command}</div>
              <div><b>Auto command:</b> ${data.auto_command}</div>
              <div><b>Lane status:</b> ${data.lane_status}</div>
            `;
          } catch (e) {
            document.getElementById("statusBox").innerHTML = "Status unavailable";
          }
        }

        setInterval(refreshStatus, 300);
        refreshStatus();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/status")
async def status():
    manual_command = "stop"
    for k in controls:
        if controls[k]:
            manual_command = k
            break

    return {
        "controls": controls,
        "manual_command": manual_command,
        "autonomous": autonomous_enabled,
        "auto_command": auto_command,
        "lane_status": lane_status
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
