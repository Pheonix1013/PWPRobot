from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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



# GET /status
@app.get("/status")
async def status():
    return controls
