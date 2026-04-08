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

LANE_WIDTH_ESTIMATE = 260
CENTER_DEADBAND = 35
HARD_TURN_THRESHOLD = 85

# -------- NEW IMPROVEMENTS --------

LANE_MEMORY_FRAMES = 8
STEERING_SMOOTHING = 0.7

last_lane_center = None
frames_since_lane = 0
last_error = 0

# ----------------------------------


def get_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur,50,150)

    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)

    return edges


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

    fit = np.polyfit(np.array(ys), np.array(xs), 1)
    return fit


def line_points_from_fit(fit, y1, y2, w):
    if fit is None:
        return None

    m, b = fit

    x1 = int(m*y1 + b)
    x2 = int(m*y2 + b)

    x1 = max(0,min(w-1,x1))
    x2 = max(0,min(w-1,x2))

    return (x1,y1,x2,y2)


def draw_line(img, fit, color):
    if fit is None:
        return

    h,w = img.shape[:2]

    p = line_points_from_fit(fit,h-1,0,w)

    if p is None:
        return

    x1,y1,x2,y2 = p

    cv2.line(img,(x1,y1),(x2,y2),color,4)


def classify_lines(lines, w):
    left_lines=[]
    right_lines=[]

    if lines is None:
        return left_lines,right_lines

    for l in lines:

        x1,y1,x2,y2 = l[0]

        if x2==x1:
            continue

        slope=(y2-y1)/(x2-x1)
        mid_x=(x1+x2)/2

        if abs(slope)<0.45:
            continue

        if slope<0 and mid_x<w*0.7:
            left_lines.append((x1,y1,x2,y2))

        elif slope>0 and mid_x>w*0.3:
            right_lines.append((x1,y1,x2,y2))

    return left_lines,right_lines


def compute_auto_command(left_fit,right_fit,w,h):

    global lane_status
    global last_lane_center
    global frames_since_lane
    global last_error

    y_look=int(h*0.82)
    frame_center=w//2

    x_left=None
    x_right=None
    lane_center=None
    mode=None

    if left_fit is not None:
        x_left=int(left_fit[0]*y_look+left_fit[1])

    if right_fit is not None:
        x_right=int(right_fit[0]*y_look+right_fit[1])

    if x_left is not None and x_right is not None:

        lane_center=(x_left+x_right)//2
        mode="Both lanes"
        frames_since_lane=0

    elif x_left is not None:

        lane_center=x_left+LANE_WIDTH_ESTIMATE//2
        mode="Left only"
        frames_since_lane=0

    elif x_right is not None:

        lane_center=x_right-LANE_WIDTH_ESTIMATE//2
        mode="Right only"
        frames_since_lane=0

    else:

        frames_since_lane+=1

        if last_lane_center is not None and frames_since_lane<=LANE_MEMORY_FRAMES:
            lane_center=last_lane_center
            mode="Lane memory"
        else:
            lane_status="No lanes"
            return "stop",None,None


    lane_center=max(0,min(w-1,lane_center))

    last_lane_center=lane_center

    error=lane_center-frame_center

    # smoothing
    error=int(STEERING_SMOOTHING*last_error+(1-STEERING_SMOOTHING)*error)
    last_error=error

    lane_status=f"{mode} | err={error}"

    if abs(error)<=CENTER_DEADBAND:
        return "forward",lane_center,error

    if error<-HARD_TURN_THRESHOLD:
        return "left",lane_center,error

    if error>HARD_TURN_THRESHOLD:
        return "right",lane_center,error

    if error<0:
        return "left",lane_center,error

    return "right",lane_center,error


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

    left_lines,right_lines=classify_lines(lines,w)

    left_fit=average_line(left_lines)
    right_fit=average_line(right_lines)

    draw_line(frame,left_fit,(0,0,255))
    draw_line(frame,right_fit,(255,0,0))

    cmd,lane_center,error=compute_auto_command(left_fit,right_fit,w,h)

    frame_center=w//2
    y_look=int(h*0.82)

    cv2.line(frame,(frame_center,0),(frame_center,h),(255,255,0),2)
    cv2.line(frame,(0,y_look),(w,y_look),(255,255,0),2)

    if lane_center is not None:
        cv2.circle(frame,(lane_center,y_look),8,(0,255,0),-1)

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

    if not ok:
        return {"status":"encode failed"}

    latest_frame=jpeg.tobytes()

    return {"status":"ok","auto_command":auto_command}


@app.get("/video_feed")
async def video_feed():

    async def stream():

        while True:

            if latest_frame:

                yield(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"+
                    latest_frame+
                    b"\r\n"
                )

            await asyncio.sleep(0.03)

    return StreamingResponse(stream(),media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
async def status():

    manual_command="stop"

    for k in controls:
        if controls[k]:
            manual_command=k
            break

    return{
        "controls":controls,
        "manual_command":manual_command,
        "autonomous":autonomous_enabled,
        "auto_command":auto_command,
        "lane_status":lane_status
    }
