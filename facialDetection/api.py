from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import cv2
import numpy as np

app = FastAPI()

# --- Global State ---
latest_frame = None
latest_frame_raw = None
autonomous_enabled = False
auto_command = "stop"
lane_status = "No lanes"
marvin_detected = False  # New flag for the popup

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controls = {"forward": False, "backward": False, "left": False, "right": False}

# --- Marvin SIFT Setup ---
sift = cv2.SIFT_create()
marvin_template = cv2.imread('marvin_template.png', cv2.IMREAD_GRAYSCALE)
if marvin_template is not None:
    kp_t, des_t = sift.detectAndCompute(marvin_template, None)
else:
    kp_t, des_t = None, None

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

def detect_marvin_logic(frame):
    if des_t is None: return frame, False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = sift.detectAndCompute(gray, None)
    if des_f is None or len(des_f) < 10: return frame, False
    matches = flann.knnMatch(des_t, des_f, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) > 12:
        src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = marvin_template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 4, cv2.LINE_AA)
            return frame, True
    return frame, False

def process_frame(frame):
    global auto_command, autonomous_enabled, lane_status, marvin_detected
    frame = cv2.resize(frame, (640, 480))
    
    # 1. Detect Marvin
    frame, marvin_detected = detect_marvin_logic(frame)
    
    # 2. Add your original Lane/Stop Line logic here
    # (Simplified for the template, keep your original get_mask/classify_lines calls)
    lane_status = "Scanning..." if not marvin_detected else "ENTITY DETECTED"
    
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
    if frame is not None:
        processed = process_frame(frame)
        ok, jpeg = cv2.imencode(".jpg", processed)
        latest_frame = jpeg.tobytes()
    return {"status": "ok", "auto_command": auto_command}

@app.get("/status")
async def status():
    manual_command = "stop"
    for k in controls:
        if controls[k]: manual_command = k; break
    return {
        "controls": controls,
        "manual_command": manual_command,
        "autonomous": autonomous_enabled,
        "auto_command": auto_command,
        "lane_status": lane_status,
        "marvin_detected": marvin_detected
    }

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

@app.post("/{direction}")
async def move(direction: str):
    global autonomous_enabled, auto_command
    if direction in controls:
        autonomous_enabled = False
        auto_command = "stop"
        for k in controls: controls[k] = False
        controls[direction] = True
    return {"status": "ok"}

@app.post("/stop")
async def stop():
    global autonomous_enabled, auto_command
    autonomous_enabled = False
    auto_command = "stop"
    for k in controls: controls[k] = False
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as f: return f.read()

@app.post("/autonomous/start")
async def autonomous_start():
    global autonomous_enabled; autonomous_enabled = True
    return {"autonomous": True}

@app.post("/autonomous/stop")
async def autonomous_stop():
    global autonomous_enabled; autonomous_enabled = False
    return {"autonomous": False}
