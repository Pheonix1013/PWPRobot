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

# remember which side we last saw
last_lane_seen = None

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
    return {"message": "All movements stopped"}


@app.post("/autonomous/start")
async def autonomous_start():
    global autonomous_enabled
    for k in controls:
        controls[k] = False
    autonomous_enabled = True
    return {"autonomous": True}


@app.post("/autonomous/stop")
async def autonomous_stop():
    global autonomous_enabled, auto_command
    autonomous_enabled = False
    auto_command = "stop"
    return {"autonomous": False}


def get_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    return edges


def average_line(lines):
    if not lines:
        return None

    xs=[]
    ys=[]

    for x1,y1,x2,y2 in lines:
        xs.extend([x1,x2])
        ys.extend([y1,y2])

    if len(xs)<2:
        return None

    fit=np.polyfit(np.array(ys,dtype=np.float32),
                   np.array(xs,dtype=np.float32),1)

    return fit[0],fit[1]


def classify_lines(lines,w,h):
    left_lines=[]
    right_lines=[]

    if lines is None:
        return left_lines,right_lines

    for l in lines:
        x1,y1,x2,y2=l[0]

        if x2==x1:
            continue

        slope=(y2-y1)/(x2-x1)
        mid_x=(x1+x2)/2

        if abs(slope)<0.4:
            continue

        if slope<0 and mid_x<w*0.7:
            left_lines.append((x1,y1,x2,y2))

        elif slope>0 and mid_x>w*0.3:
            right_lines.append((x1,y1,x2,y2))

    return left_lines,right_lines


def compute_auto_command(left_fit,right_fit,w,h):
    global lane_status,last_lane_seen

    y_look=int(h*0.82)
    frame_center=w//2

    x_left=None
    x_right=None

    if left_fit is not None:
        x_left=int(left_fit[0]*y_look+left_fit[1])
        last_lane_seen="left"

    if right_fit is not None:
        x_right=int(right_fit[0]*y_look+right_fit[1])
        last_lane_seen="right"

    # BOTH LANES
    if x_left is not None and x_right is not None:

        lane_center=(x_left+x_right)//2
        error=lane_center-frame_center

        lane_status=f"Both lanes err={error}"

        if abs(error)<=CENTER_DEADBAND:
            return "forward",lane_center,error

        if error<0:
            return "left",lane_center,error

        return "right",lane_center,error

    # LEFT LOST
    if x_left is None and x_right is not None:
        lane_status="Left lane lost -> turning LEFT"
        return "left",None,None

    # RIGHT LOST
    if x_right is None and x_left is not None:
        lane_status="Right lane lost -> turning RIGHT"
        return "right",None,None

    # BOTH LOST
    if last_lane_seen=="left":
        lane_status="Both lost -> sweeping LEFT"
        return "left",None,None

    if last_lane_seen=="right":
        lane_status="Both lost -> sweeping RIGHT"
        return "right",None,None

    lane_status="No lanes"
    return "stop",None,None


def process_frame(frame):
    global auto_command

    frame=cv2.resize(frame,(640,480))

    mask=get_mask(frame)

    lines=cv2.HoughLinesP(
        mask,
        1,
        np.pi/180,
        threshold=35,
        minLineLength=30,
        maxLineGap=30
    )

    h,w=frame.shape[:2]

    left_lines,right_lines=classify_lines(lines,w,h)

    left_fit=average_line(left_lines)
    right_fit=average_line(right_lines)

    cmd,lane_center,error=compute_auto_command(left_fit,right_fit,w,h)

    auto_command=cmd

    return frame


@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame,latest_frame_raw

    data=await request.json()
    frame_b64=data.get("frame")

    if not frame_b64:
        return {"status":"no frame"}

    frame_bytes=base64.b64decode(frame_b64)
    latest_frame_raw=frame_bytes

    np_arr=np.frombuffer(frame_bytes,np.uint8)
    frame=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

    if frame is None:
        return {"status":"bad frame"}

    processed=process_frame(frame)

    ok,jpeg=cv2.imencode(".jpg",processed)

    if ok:
        latest_frame=jpeg.tobytes()

    return {"status":"ok","auto_command":auto_command}


@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            if latest_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"+
                    latest_frame+
                    b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(stream(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_raw")
async def video_feed_raw():
    async def stream():
        while True:
            if latest_frame_raw:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"+
                    latest_frame_raw+
                    b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(stream(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
async def status():
    manual_command="stop"

    for k in controls:
        if controls[k]:
            manual_command=k
            break

    return {
        "manual_command":manual_command,
        "autonomous":autonomous_enabled,
        "auto_command":auto_command,
        "lane_status":lane_status
    }
