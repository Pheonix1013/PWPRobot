from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import numpy as np
import cv2

app = FastAPI()

latest_frame = None

# Enable CORS so your GUI can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# -----------------------------
# Perspective Transform Function
# -----------------------------
def apply_perspective_transform(frame_bytes):
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return frame_bytes

    h, w = frame.shape[:2]

    src = np.float32([
        [85, 100],
        [w - 60, 100],
        [40, h - 50],
        [w - 20, h - 50]
    ])

    dst = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (w, h))

    _, buffer = cv2.imencode('.jpg', warped)
    return buffer.tobytes()


# -----------------------------
# Shared Frame Streaming Generator
# -----------------------------
async def frame_stream(transform=False):
    global latest_frame

    while True:
        if latest_frame:
            raw_bytes = base64.b64decode(latest_frame)

            if transform:
                frame_bytes = apply_perspective_transform(raw_bytes)
            else:
                frame_bytes = raw_bytes

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

        await asyncio.sleep(0.05)  # smoother streaming


# -----------------------------
# Video Endpoints
# -----------------------------

# Perspective transformed feed
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        frame_stream(transform=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Raw feed
@app.get("/video_feed_raw")
async def video_feed_raw():
    return StreamingResponse(
        frame_stream(transform=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# Upload Frame Endpoint
# -----------------------------
@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame
    data = await request.json()
    latest_frame = data.get("frame")
    return {"status": "ok"}


# -----------------------------
# Robot Control Endpoints
# -----------------------------

@app.post("/stop")
async def stop():
    for key in controls:
        controls[key] = False
    return {"message": "All movements stopped"}


@app.post("/{direction}")
async def move(direction: str):
    if direction not in controls:
        return {"error": "Invalid direction"}

    # Reset all controls
    for key in controls:
        controls[key] = False

    # Toggle selected direction
    controls[direction] = True

    return {direction: controls[direction]}


@app.get("/status")
async def status():
    return controls
