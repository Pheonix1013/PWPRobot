from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import io
from PIL import Image
import cv2
import numpy as np

app = FastAPI()

latest_frame = None
latest_frame_raw = None

# Enable CORS so your GUI can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robot controls state
controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}
# POST /stop
@app.post("/stop")
async def stop():
    for key in controls:
        controls[key] = False
    return {"message": "All movements stopped"}

def process_frame(frame):
    """
    Runs lane detection + center line overlay on a frame.
    Returns processed OpenCV frame.
    """

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # --- Crop region (road area) ---
    CROP_X, CROP_Y, CROP_W, CROP_H = 80, 60, 480, 360
    cropped = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W].copy()

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Threshold mask
    mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 7
    )

    # Morph cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort largest 2 contours (lane lines)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    if len(contours) == 2:
        cntA, cntB = contours

        # Sort left vs right
        def mean_x(cnt):
            return cnt.reshape(-1, 2)[:, 0].mean()

        if mean_x(cntA) < mean_x(cntB):
            left_cnt, right_cnt = cntA, cntB
        else:
            left_cnt, right_cnt = cntB, cntA

        # Draw contours
        cv2.drawContours(cropped, [left_cnt], -1, (0, 0, 255), 2)
        cv2.drawContours(cropped, [right_cnt], -1, (255, 0, 0), 2)

        # Fit lines
        def fit_line(cnt):
            pts = cnt.reshape(-1, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]
            m, b = np.polyfit(ys, xs, 1)
            return m, b

        mL, bL = fit_line(left_cnt)
        mR, bR = fit_line(right_cnt)

        # Centerline
        ys = np.linspace(0, cropped.shape[0]-1, 50)
        midpoints = []

        for yy in ys:
            xL = mL * yy + bL
            xR = mR * yy + bR
            midpoints.append(((xL+xR)/2, yy))

        midpoints = np.array(midpoints)
        mC, bC = np.polyfit(midpoints[:, 1], midpoints[:, 0], 1)

        # Draw center line
        x_top = int(mC*0 + bC)
        x_bot = int(mC*(cropped.shape[0]-1) + bC)

        cv2.line(
            cropped,
            (x_top, 0),
            (x_bot, cropped.shape[0]-1),
            (0, 255, 0),
            3
        )

    # Put cropped back into main frame
    frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W] = cropped

    return frame

@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, latest_frame_raw

    data = await request.json()
    frame_b64 = data.get("frame")

    if not frame_b64:
        return {"status": "no frame"}

    # Decode JPEG bytes (RAW JPEG from robot)
    frame_bytes = base64.b64decode(frame_b64)

    latest_frame_raw = frame_bytes

    # Convert bytes → OpenCV frame
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run lane detection processing
    processed = process_frame(frame)

    # Convert processed frame back → JPEG bytes
    _, jpeg = cv2.imencode(".jpg", processed)

    latest_frame = jpeg.tobytes()

    return {"status": "ok"}

@app.get("/video_feed")
async def video_feed():
    async def frame_stream():
        while True:
            if latest_frame:
                frame_bytes = base64.b64decode(latest_frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            await asyncio.sleep(0.1)
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_raw")
async def video_feed_raw():
    async def frame_stream():
        while True:
            if latest_frame_raw:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + latest_frame_raw +
                       b"\r\n")
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    html = """
    <html>
      <head>
        <title>Robot Web GUI</title>
        <style>
          body { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; height: 100vh; margin: 0; }
          .quadrant { border: 1px solid #444; display: flex; justify-content: center; align-items: center; font-size: 1.2rem; }
        </style>
      </head>
      <body>
        <div class="quadrant" id="top-left">
          <img src="/video_feed" width="480" height="320">
        </div>
        <div class="quadrant" id="top-right">Controls</div>
        <div class="quadrant" id="bottom-left">Sensors</div>
        <div class="quadrant" id="bottom-right">Status</div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

# GET /status
@app.get("/status")
async def status():
    return controls

# POST /<direction>
@app.post("/{direction}")
async def move(direction: str):
    # Reset all controls
    for key in controls:
        controls[key] = False

    # Toggle selected direction
    controls[direction] = not controls[direction]

    return {direction: controls[direction]}

