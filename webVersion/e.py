from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import sqlite3
import cv2
import numpy as np
import threading
import asyncio
from datetime import datetime

# ---------------- DB (sqlite) ----------------
# store users locally, same as you used before
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

class User(BaseModel):
    username: str
    password: str

# ---------------- FastAPI app ----------------
app = FastAPI()

# allow requests from any origin (your GUI will request locally)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Robot state ----------------
controls = {"forward": False, "backward": False, "left": False, "right": False}

# ---------------- Camera + Processing params ----------------
FRAME_W = 640
FRAME_H = 480

# stoering the variables (so i can fine tune them for testing)
BLUR_K = 9
TH_BLOCK = 51
TH_C = 7
MORPH_K = 7

# more variables - for contour
MIN_ARCLEN = 150.0
MIN_AREA = 200

# number of samples that we gonna take for the line of best fit
NUM_SAMPLES = 60
SMOOTH_WIN = 7

# drawing
CROP_X, CROP_Y, CROP_W, CROP_H = 160, 120, 320, 240
LINE_THICK = 3

# ---------------- Globals for frame sharing ----------------
latest_frame = None        # will hold latest processed jpeg bytes
latest_frame_raw = None
latest_frame_lock = threading.Lock()

LOG_FILE = "user_log.txt"
log_lock = threading.Lock()

def log_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")


# ---------------- OpenCV camera init ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# ---------------- Helper functions (with docstrings) ----------------

def pleaseWork(contour_pts):
    """
    Fit a line to contour points and return best vertical/horizontal line.
    (tries both 'vertical' fit xs vs ys and 'horizontal' fit ys vs xs, picks the smaller error)

    Parameters:
    contour_pts : numpy.ndarray of shape (N,1,2)

    Return:
    tuple: (mode, slope m, intercept b, error)
           mode is 'v' (x = m*y + b) or 'h' (y = m*x + b)
    """
    pts = contour_pts.reshape(-1, 2).astype(float)
    xs = pts[:, 0]
    ys = pts[:, 1]
    best = ("v", 0, 0, 1e12)
    try:
        m_v, b_v = np.polyfit(ys, xs, 1)
        err_v = np.mean((xs - (m_v*ys + b_v))**2)
        best = ("v", m_v, b_v, err_v)
    except Exception:
        pass
    try:
        m_h, b_h = np.polyfit(xs, ys, 1)
        err_h = np.mean((ys - (m_h*xs + b_h))**2)
        if err_h < best[3]:
            best = ("h", m_h, b_h, err_h)
    except Exception:
        pass
    return best

def checkIfMoving(a, n):
    """
    Smooth the array using moving average so lines arent jittery.

    Parameters:
    a : 1D numpy array
    n : int window size for moving average

    Return:
    numpy array (smoothed)
    """
    if len(a) < n:
        return a
    return np.convolve(a, np.ones(n)/n, mode='same')

def extendLines(mode, m, b, width, height):
    """
    Extend a fitted line (mode 'v' or 'h') to the rectangle edges and return two points.

    Parameters:
    mode : 'v' or 'h'
    m, b : slope and intercept
    width, height : dimensions of the image/crop

    Return:
    p1, p2 : (x,y) integer tuples
    """
    if mode == "v":
        x0 = m*0 + b
        x1 = m*(height-1) + b
        p1 = (int(round(x0)), 0)
        p2 = (int(round(x1)), height-1)
    else:
        y0 = m*0 + b
        y1 = m*(width-1) + b
        p1 = (0, int(round(y0)))
        p2 = (width-1, int(round(y1)))
    return p1, p2

def process_and_update_frame():
    """
    Main OpenCV loop: capture frames, run your line-detection + drawing,
    encode to JPEG, and put into latest_frame (thread-safe).
    Runs in a background thread.
    """
    global latest_frame, latest_frame_raw, cap

    # if camera failed to open, bail out quietly
    if not cap.isOpened():
        print("ayyy camera not opened, check your camera index")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # skip if frame not grabbed

        # ensure frame is expected size (some cameras ignore set())
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        # save RAW frame (no drawings)
        ret_raw, jpeg_raw = cv2.imencode('.jpg', frame)
        if ret_raw:
    	    with latest_frame_lock:
                latest_frame_raw = jpeg_raw.tobytes()


        # --------- cropping + processing -----------
        x2 = min(CROP_X + CROP_W, FRAME_W)
        y2 = min(CROP_Y + CROP_H, FRAME_H)
        cropped = frame[CROP_Y:y2, CROP_X:x2].copy()
        h = cropped.shape[0]
        w = cropped.shape[1]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, TH_BLOCK, TH_C)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good = []
        for cnt in contours:
            al = cv2.arcLength(cnt, closed=False)
            area = cv2.contourArea(cnt)
            if al >= MIN_ARCLEN and area >= MIN_AREA:
                good.append((al, cnt))

        # fallback: if we don't have 2 big contours, take top few by area (like before)
        if len(good) < 2:
            contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
            for cnt in contours_sorted[:3]:
                if cv2.contourArea(cnt) >= 50:
                    good.append((cv2.arcLength(cnt, False), cnt))

        good = sorted(good, key=lambda x: x[0], reverse=True)[:2]

        fullframe_lines = []

        if len(good) >= 2:
            cntA = good[0][1]
            cntB = good[1][1]

            def mean_x(cnt):
                pts = cnt.reshape(-1,2)
                return pts[:,0].mean()

            if mean_x(cntA) <= mean_x(cntB):
                left_cnt, right_cnt = cntA, cntB
            else:
                left_cnt, right_cnt = cntB, cntA

            modeL, mL, bL, _ = pleaseWork(left_cnt)
            modeR, mR, bR, _ = pleaseWork(right_cnt)

            # draw the contours in the cropped frame only not main frame yet
            cv2.drawContours(cropped, [left_cnt], -1, (0,0,255), 2)
            cv2.drawContours(cropped, [right_cnt], -1, (255,0,0), 2)

            # get midpoints to draw the best fit line using averaging of the slope
            midpoints = []

            if modeL == "v" and modeR == "v":
                ys = np.linspace(0, h-1, NUM_SAMPLES)
                for yy in ys:
                    xL = mL*yy + bL
                    xR = mR*yy + bR
                    midpoints.append(((xL+xR)/2.0, yy))
            elif modeL == "h" and modeR == "h":
                xs = np.linspace(0, w-1, NUM_SAMPLES)
                for xx in xs:
                    yL = mL*xx + bL
                    yR = mR*xx + bR
                    midpoints.append((xx, (yL+yR)/2.0))
            else:
                ts = np.linspace(0.0, 1.0, NUM_SAMPLES)
                for t in ts:
                    xx = t*(w-1)
                    yy = t*(h-1)
                    if modeL == "v":
                        xL = mL*yy + bL; yL = yy
                    else:
                        yL = mL*xx + bL; xL = xx
                    if modeR == "v":
                        xR = mR*yy + bR; yR = yy
                    else:
                        yR = mR*xx + bR; xR = xx
                    midpoints.append(((xL+xR)/2.0, (yL+yR)/2.0))

            if len(midpoints) >= 3:
                pts = np.array(midpoints)
                xs = checkIfMoving(pts[:,0], SMOOTH_WIN)
                ys = checkIfMoving(pts[:,1], SMOOTH_WIN)
                mid_s = np.vstack((xs, ys)).T

                # bunch of try/except so code dont break
                err_v = err_h = 1e12
                try:
                    mcv, bcv = np.polyfit(mid_s[:,1], mid_s[:,0], 1)
                    err_v = np.mean((mid_s[:,0] - (mcv*mid_s[:,1] + bcv))**2)
                except Exception:
                    pass
                try:
                    mch, bch = np.polyfit(mid_s[:,0], mid_s[:,1], 1)
                    err_h = np.mean((mid_s[:,1] - (mch*mid_s[:,0] + bch))**2)
                except Exception:
                    pass

                if err_v <= err_h:
                    p_top = (int(round(mcv*0 + bcv)), 0)
                    p_bot = (int(round(mcv*(h-1) + bcv)), h-1)
                else:
                    p_top = (0, int(round(mch*0 + bch)))
                    p_bot = (w-1, int(round(mch*(w-1) + bch)))

                def clip(p):
                    return (int(max(0,min(w-1,p[0]))), int(max(0,min(h-1,p[1]))))

                p1 = clip(p_top)
                p2 = clip(p_bot)

                # draw center line in cropped
                cv2.line(cropped, p1, p2, (0,255,0), LINE_THICK)

                # add the cropped frame x and y to the lines so that the lines don't show in the wrong place for the main frame
                p1_abs = (p1[0] + CROP_X, p1[1] + CROP_Y)
                p2_abs = (p2[0] + CROP_X, p2[1] + CROP_Y)
                fullframe_lines.append((p1_abs, p2_abs, (0,255,0)))

            # draw left (red) line
            pL1, pL2 = extendLines(modeL, mL, bL, w, h)
            cv2.line(cropped, pL1, pL2, (0,0,255), LINE_THICK)
            fullframe_lines.append(((pL1[0]+CROP_X, pL1[1]+CROP_Y),
                                    (pL2[0]+CROP_X, pL2[1]+CROP_Y),
                                    (0,0,255)))

            # draw right (blue) line
            pR1, pR2 = extendLines(modeR, mR, bR, w, h)
            cv2.line(cropped, pR1, pR2, (255,0,0), LINE_THICK)
            fullframe_lines.append(((pR1[0]+CROP_X, pR1[1]+CROP_Y),
                                    (pR2[0]+CROP_X, pR2[1]+CROP_Y),
                                    (255,0,0)))

        # draw crop rectangle box
        cv2.rectangle(frame, (CROP_X, CROP_Y), (x2, y2), (0,0,255), 2)

        # draw all lines on the main frame instead of just crop
        for (pa, pb, col) in fullframe_lines:
            cv2.line(frame, pa, pb, col, LINE_THICK)

        # --------- prepare jpeg and store to latest_frame (thread-safe) ---------
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if ret2:
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()

        # small sleep to avoid hogging CPU - this controls frame rate approx
        # if your camera gives 30fps you can reduce or remove this
        # but keep a tiny sleep to let server threads breathe
        cv2.waitKey(1)

# start background processing thread
processing_thread = threading.Thread(target=process_and_update_frame, daemon=True)
processing_thread.start()

# ---------------- Streaming endpoint ----------------
@app.get("/video_feed")
async def video_feed():
    """
    MJPEG stream of latest processed frames.
    """
    async def frame_stream():
        while True:
            data = None
            with latest_frame_lock:
                if latest_frame:
                    data = latest_frame
            if data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       data + b"\r\n")
            await asyncio.sleep(0.03)  # ~30 fps
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_raw")
async def video_feed_raw():
    async def frame_stream():
        while True:
            data = None
            with latest_frame_lock:
                if latest_frame_raw:
                    data = latest_frame_raw
            if data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       data + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ---------------- Login / Register endpoints ----------------
@app.post("/register")
async def register(user: User):
    """
    Register user in sqlite. returns error if exists.
    """
    try:
        cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", (user.username, user.password))
        conn.commit()
        log_event(f"REGISTER | user={user.username}")
        return {"message": "Registration successful"}
    except Exception:
        raise HTTPException(status_code=400, detail="Username already exists or invalid")

@app.post("/login")
async def login(user: User):
    """
    Verify user credentials.
    """
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user.username, user.password))
    if cursor.fetchone():
        log_event(f"LOGIN success | user={user.username}")
        return {"message": "Login successful"}

    log_event(f"LOGIN failed | user={user.username}")
    raise HTTPException(status_code=401, detail="Invalid username/password")

# ---------------- Controls ----------------
@app.get("/status")
async def status():
    """
    Return current control state dict.
    """
    return controls

@app.post("/stop")
async def stop():
    """
    Reset movement controls.
    """
    for k in controls:
        controls[k] = False

    log_event("CONTROL | stop")
    return {"message": "All movements stopped"}

@app.post("/{direction}")
async def move(direction: str):
    """
    Set a movement direction (forward/backward/left/right).
    """
    if direction not in controls:
        raise HTTPException(status_code=400, detail="Invalid direction")
    for k in controls:
        controls[k] = False
    controls[direction] = True

    log_event(f"CONTROL | direction={direction}")
    return {direction: True}

# ---------------- Serve the GUI HTML ----------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the web GUI: login screen -> on success shows 4 quadrants fed from /video_feed
    """
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>PWP Robot Control</title>
<style>
  body { margin: 0; font-family: Arial; }
  #login-screen { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; }
  #app-screen { display:none; height:100vh; padding: 10px; box-sizing: border-box; }
  table { width: 100%; height: 100%; border-collapse: collapse; }
  td { border: 1px solid black; text-align:center; vertical-align:middle; }
  img { display:block; margin: 0 auto; }
</style>
</head>
<body>

<!-- LOGIN SCREEN -->
<div id="login-screen">
  <h1>Robot Control Login</h1>
  <input id="username" placeholder="Username" />
  <input id="password" placeholder="Password" type="password" />
  <br>
  <button onclick="registerUser()">Register</button>
  <button onclick="loginUser()">Login</button>
  <p id="login-msg"></p>
</div>

<!-- MAIN GUI -->
<div id="app-screen">
  <table>
    <tr height="50%">
      <td width="50%">
        <img src="/video_feed" width="480" height="320" alt="video"/>
      </td>

      <td width="50%">
        <table style="margin: 0 auto;">
          <tr align="center">
            <td></td>
            <td><button onclick="sendCommand('forward')">&#8593;</button></td>
            <td></td>
          </tr>
          <tr align="center">
            <td><button onclick="sendCommand('left')">&#8592;</button></td>
            <td><button onclick="stopMotor()">&#9632;</button></td>
            <td><button onclick="sendCommand('right')">&#8594;</button></td>
          </tr>
          <tr align="center">
            <td></td>
            <td><button onclick="sendCommand('backward')">&#8595;</button></td>
            <td></td>
          </tr>
        </table>
      </td>
    </tr>

    <tr height="50%">
      <td width="50%">
        <img src="/video_feed_raw" width="480" height="320" alt="video"/>
      </td>
      <td width="50%">
        <h3>Console Log</h3>
        <pre id="console-log" style="height:300px; overflow:auto; border:1px solid black;"></pre>
      </td>
    </tr>
  </table>
</div>

<script>
const API_BASE = "http://127.0.0.1:5000";

/* ---------- LOGIN & REGISTER ---------- */
async function registerUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  try {
    const res = await fetch(API_BASE + "/register", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
  } catch (e) {
    document.getElementById("login-msg").innerText = "Network error";
  }
}

async function loginUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  try {
    const res = await fetch(API_BASE + "/login", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
    if (res.ok) {
      document.getElementById("login-screen").style.display = "none";
      document.getElementById("app-screen").style.display = "block";
    }
  } catch (e) {
    document.getElementById("login-msg").innerText = "Network error";
  }
}

/* ---------- CONTROLS ---------- */
async function sendCommand(direction) {
  log("POST /" + direction);
  try {
    const res = await fetch(API_BASE + "/" + direction, {method: "POST"});
    const data = await res.json();
    log("Response: " + JSON.stringify(data));
  } catch (e) {
    log("Network error sending " + direction);
  }
}

async function stopMotor() {
  log("POST /stop");
  try {
    const res = await fetch(API_BASE + "/stop", {method: "POST"});
    const data = await res.json();
    log("Response: " + JSON.stringify(data));
  } catch (e) {
    log("Network error sending stop");
  }
}

/* ---------- LOG ---------- */
function log(msg) {
  const box = document.getElementById("console-log");
  box.textContent += msg + "\\n";
  box.scrollTop = box.scrollHeight;
}
</script>

</body>
</html>
""")

# ---------------- Shutdown cleanup ----------------
@app.on_event("shutdown")
def shutdown_event():
    try:
        cap.release()
    except Exception:
        pass
