from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import base64
import sqlite3

app = FastAPI()

# -----------------------------
# DATABASE (SQLite)
# -----------------------------
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

# -----------------------------
# CORS SETTINGS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ROBOT STATE + VIDEO
# -----------------------------
latest_frame = None

controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}

@app.post("/stop")
async def stop():
    for key in controls:
        controls[key] = False
    return {"message": "All movements stopped"}


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
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       frame_bytes + b"\r\n")
            await asyncio.sleep(0.1)

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# LOGIN + REGISTER
# -----------------------------
@app.post("/register")
async def register(user: User):
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (user.username, user.password)
        )
        conn.commit()
        return {"message": "Registration successful!"}
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )


@app.post("/login")
async def login(user: User):
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (user.username, user.password)
    )
    found = cursor.fetchone()

    if not found:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    return {"message": "Login successful"}


# -----------------------------
# ROBOT MOVEMENT
# -----------------------------
@app.get("/status")
async def status():
    return controls


@app.post("/{direction}")
async def move(direction: str):
    if direction not in controls:
        raise HTTPException(status_code=400, detail="Invalid direction")

    for key in controls:
        controls[key] = False

    controls[direction] = True

    return {direction: True}


# -----------------------------
# DEFAULT PAGE
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return "<h1>Robot API Running</h1>"
