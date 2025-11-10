from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import io
from PIL import Image
import cv2


app = FastAPI()

latest_frame = None

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

# POST /<direction>
@app.post("/{direction}")
async def move(direction: str):
    # Reset all controls
    for key in controls:
        controls[key] = False

    # Toggle selected direction
    controls[direction] = not controls[direction]

    return {direction: controls[direction]}

@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame
    data = await request.json()
    latest_frame = data.get("frame")
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
