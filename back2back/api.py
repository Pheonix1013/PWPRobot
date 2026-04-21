from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import threading
import time
import base64
import importlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

md = importlib.import_module("motorDriver")
pp = importlib.import_module("processing_parallel")

raw_frame = None
proc_frame = None
frame_lock = threading.Lock()

auto_on = False
auto_thread = None

last_manual = "stop"
last_auto = "idle"
lane_state = "none"
last_err = 0.0
tape_seen = False
tape_seen_at = 0.0

BASE_SPEED = 115
TURN_SPEED = 130
TURN_TIME = 0.55
FORWARD_AFTER_TURN_TIME = 0.9
KP = 0.55

LEFT_TRIM = 0.86
RIGHT_TRIM = 1.00

TAPE_CONFIRM_TIME = 0.10
BOTTOM_BAND_FRAC = 0.34
MIN_TAPE_WIDTH_FRAC = 0.38
MIN_TAPE_HEIGHT = 7
MIN_TAPE_AREA = 1800


def clamp(v, lo=0, hi=255):
    return int(max(lo, min(hi, v)))


def _run_motor(idx, direction, speed):
    if hasattr(md, "MotorRun"):
        md.MotorRun(idx, direction, clamp(speed))
        return True
    return False


def _stop_all():
    if hasattr(md, "stop_all"):
        md.stop_all()
        return
    if hasattr(md, "Stop"):
        md.Stop()
        return
    if hasattr(md, "MotorStop"):
        try:
            md.MotorStop(0)
            md.MotorStop(1)
            return
        except:
            pass
    _run_motor(0, "forward", 0)
    _run_motor(1, "forward", 0)


def _tank(left_speed, right_speed):
    ls = clamp(abs(left_speed) * LEFT_TRIM)
    rs = clamp(abs(right_speed) * RIGHT_TRIM)

    ld = "forward" if left_speed >= 0 else "backward"
    rd = "forward" if right_speed >= 0 else "backward"

    if hasattr(md, "set_motors"):
        try:
            md.set_motors(int(left_speed * LEFT_TRIM), int(right_speed * RIGHT_TRIM))
            return
        except:
            pass

    ok0 = _run_motor(0, ld, ls)
    ok1 = _run_motor(1, rd, rs)

    if not (ok0 and ok1):
        raise RuntimeError("motorDriver.py needs MotorRun() or set_motors().")


def go_forward(speed=BASE_SPEED):
    _tank(speed, speed)


def go_backward(speed=BASE_SPEED):
    _tank(-speed, -speed)


def pivot_left(speed=TURN_SPEED):
    _tank(-speed, speed)


def pivot_right(speed=TURN_SPEED):
    _tank(speed, -speed)


def steer_drive(base, steer):
    left = base - steer
    right = base + steer
    _tank(left, right)


def mjpeg_gen(which="proc"):
    while True:
        with frame_lock:
            f = proc_frame.copy() if which == "proc" and proc_frame is not None else None
            if which == "raw":
                f = raw_frame.copy() if raw_frame is not None else None

        if f is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No frame", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", blank)
        else:
            ok, buf = cv2.imencode(".jpg", f)

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )
        time.sleep(0.03)


def decode_b64_image(s):
    if "," in s:
        s = s.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def colored_horizontal_tape(frame):
    h, w = frame.shape[:2]
    y0 = int(h * (1.0 - BOTTOM_BAND_FRAC))
    roi = frame[y0:h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    strong_color = cv2.inRange(hsv, (0, 70, 60), (179, 255, 255))

    blue1 = cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))
    purple1 = cv2.inRange(hsv, (135, 70, 50), (165, 255, 255))
    pink1 = cv2.inRange(hsv, (165, 70, 60), (179, 255, 255))
    pink2 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    yellow1 = cv2.inRange(hsv, (15, 80, 70), (40, 255, 255))

    mask = blue1 | purple1 | pink1 | pink2 | yellow1
    mask = cv2.bitwise_and(mask, strong_color)

    nongray = ((sat > 70) & (val > 60)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, nongray)

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < MIN_TAPE_AREA:
            continue
        if ww < int(w * MIN_TAPE_WIDTH_FRAC):
            continue
        if hh < MIN_TAPE_HEIGHT:
            continue
        if ww < 3.2 * hh:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, ww, hh)

    dbg = frame.copy()
    cv2.rectangle(dbg, (0, y0), (w - 1, h - 1), (0, 100, 255), 2)

    if best is not None:
        x, y, ww, hh = best
        cv2.rectangle(dbg, (x, y + y0), (x + ww, y + hh + y0), (0, 255, 255), 3)
        cv2.putText(dbg, "Tape", (x, max(25, y + y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return True, dbg

    return False, dbg


def parse_processing_result(r):
    out_img = None
    info = {}

    if isinstance(r, tuple):
        if len(r) >= 1:
            out_img = r[0]
        if len(r) >= 2 and isinstance(r[1], dict):
            info = r[1]
    elif isinstance(r, dict):
        info = r
    elif isinstance(r, np.ndarray):
        out_img = r

    return out_img, info


def do_processing(frame):
    if hasattr(pp, "process_frame"):
        r = pp.process_frame(frame.copy())
        out_img, info = parse_processing_result(r)
    else:
        out_img, info = frame.copy(), {}

    if out_img is None:
        out_img = frame.copy()

    err = 0.0
    status = "none"
    hard_turn = None

    for k in ["error", "center_error", "cx_error", "offset", "steer_error"]:
        if k in info:
            try:
                err = float(info[k])
                break
            except:
                pass

    for k in ["lane_status", "status", "lane_state"]:
        if k in info:
            status = str(info[k])
            break

    for k in ["hard_turn", "turn", "intersection_turn"]:
        if k in info and info[k] in ["left", "right"]:
            hard_turn = info[k]
            break

    if hard_turn is None:
        if "left" in status and ("hard" in status or "turn" in status):
            hard_turn = "left"
        elif "right" in status and ("hard" in status or "turn" in status):
            hard_turn = "right"

    return out_img, info, err, status, hard_turn


def auto_loop():
    global auto_on, proc_frame, lane_state, last_auto, last_err, tape_seen, tape_seen_at

    while auto_on:
        with frame_lock:
            f = None if raw_frame is None else raw_frame.copy()

        if f is None:
            time.sleep(0.02)
            continue

        out_img, info, err, status, hard_turn = do_processing(f)
        tape_hit, tape_dbg = colored_horizontal_tape(out_img)

        lane_state = status
        last_err = err

        if tape_hit:
            if not tape_seen:
                tape_seen = True
                tape_seen_at = time.time()
            elif time.time() - tape_seen_at >= TAPE_CONFIRM_TIME:
                last_auto = "stop_tape"
                _stop_all()
                auto_on = False
                with frame_lock:
                    proc_frame = tape_dbg.copy()
                break
        else:
            tape_seen = False
            tape_seen_at = 0.0

        if hard_turn == "left":
            last_auto = "hard_left"
            pivot_left(TURN_SPEED)
            t0 = time.time()
            while auto_on and time.time() - t0 < TURN_TIME:
                time.sleep(0.01)
            if not auto_on:
                break
            last_auto = "forward_after_left"
            go_forward(BASE_SPEED)
            t1 = time.time()
            while auto_on and time.time() - t1 < FORWARD_AFTER_TURN_TIME:
                time.sleep(0.01)

        elif hard_turn == "right":
            last_auto = "hard_right"
            pivot_right(TURN_SPEED)
            t0 = time.time()
            while auto_on and time.time() - t0 < TURN_TIME:
                time.sleep(0.01)
            if not auto_on:
                break
            last_auto = "forward_after_right"
            go_forward(BASE_SPEED)
            t1 = time.time()
            while auto_on and time.time() - t1 < FORWARD_AFTER_TURN_TIME:
                time.sleep(0.01)

        else:
            steer = int(KP * err)
            steer = max(-65, min(65, steer))
            steer_drive(BASE_SPEED, steer)
            last_auto = f"track:{steer}"

        show = tape_dbg.copy()
        cv2.putText(show, f"auto={auto_on}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(show, f"lane={lane_state}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(show, f"cmd={last_auto}", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(show, f"err={last_err:.2f}", (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

        with frame_lock:
            proc_frame = show

        time.sleep(0.03)

    _stop_all()
    if proc_frame is None and raw_frame is not None:
        with frame_lock:
            proc_frame = raw_frame.copy()


@app.get("/")
def root():
    return {"ok": True}


@app.get("/gui", response_class=HTMLResponse)
def gui_page():
    try:
        with open("gui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except:
        return HTMLResponse("<h1>gui.html not found</h1>", status_code=404)


@app.post("/upload_frame")
async def upload_frame(req: Request):
    global raw_frame, proc_frame

    data = await req.json()
    if "image" not in data:
        return JSONResponse({"ok": False, "error": "Missing image field"}, status_code=400)

    img = decode_b64_image(data["image"])
    if img is None:
        return JSONResponse({"ok": False, "error": "Bad image"}, status_code=400)

    out_img, info, err, status, hard_turn = do_processing(img)
    tape_hit, tape_dbg = colored_horizontal_tape(out_img)

    cv2.putText(tape_dbg, f"lane={status}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(tape_dbg, f"err={err:.2f}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)
    cv2.putText(tape_dbg, f"turn={hard_turn}", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(tape_dbg, f"tape={tape_hit}", (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    with frame_lock:
        raw_frame = img
        proc_frame = tape_dbg

    return {
        "ok": True,
        "lane_status": status,
        "error": err,
        "hard_turn": hard_turn,
        "tape": tape_hit
    }


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_gen("proc"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed_raw")
def video_feed_raw():
    return StreamingResponse(
        mjpeg_gen("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def status():
    return {
        "autonomous": auto_on,
        "manual_command": last_manual,
        "auto_command": last_auto,
        "lane_status": lane_state,
        "error": last_err,
        "tape_seen": tape_seen
    }


@app.post("/forward")
def forward():
    global last_manual
    last_manual = "forward"
    _stop_auto()
    go_forward(BASE_SPEED)
    return {"ok": True}


@app.post("/backward")
def backward():
    global last_manual
    last_manual = "backward"
    _stop_auto()
    go_backward(BASE_SPEED)
    return {"ok": True}


@app.post("/left")
def left():
    global last_manual
    last_manual = "left"
    _stop_auto()
    pivot_left(TURN_SPEED)
    return {"ok": True}


@app.post("/right")
def right():
    global last_manual
    last_manual = "right"
    _stop_auto()
    pivot_right(TURN_SPEED)
    return {"ok": True}


@app.post("/stop")
def stop():
    global last_manual
    last_manual = "stop"
    _stop_auto()
    _stop_all()
    return {"ok": True}


def _stop_auto():
    global auto_on, auto_thread, last_auto
    auto_on = False
    last_auto = "idle"
    if auto_thread is not None and auto_thread.is_alive():
        auto_thread.join(timeout=1.0)
    auto_thread = None


@app.post("/autonomous/start")
def autonomous_start():
    global auto_on, auto_thread, last_auto
    if auto_on:
        return {"ok": True, "already_running": True}

    auto_on = True
    last_auto = "starting"
    auto_thread = threading.Thread(target=auto_loop, daemon=True)
    auto_thread.start()
    return {"ok": True}


@app.post("/autonomous/stop")
def autonomous_stop():
    _stop_auto()
    _stop_all()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
