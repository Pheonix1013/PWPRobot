#import libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so the GUI can talk to it basically
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Dictionary of possible controls
controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}
#Get the bot to stop and set everything to false
@app.post("/stop")
async def stop():
    for key in controls:
        controls[key] = False
    return {"message": "All movements stopped"}

#Send post reqeusts for the directin, first set everything to false and then set the right one to true
@app.post("/{direction}")
async def move(direction: str):
    for key in controls:
        controls[key] = False
    controls[direction] = not controls[direction]
    return {direction: controls[direction]}



#Sebd get request to display the dictionary so we can trouble shoot
@app.get("/status")
async def status():
    return controls
