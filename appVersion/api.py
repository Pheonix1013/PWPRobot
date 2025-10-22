from flask import Flask, jsonify

app = Flask(__name__)

# robot controls state
controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}

@app.route("/<direction>", methods=["POST"])
def move(direction):
    """
    Create dynamic API that changes the direction
    
    Parameters:
    Direction
    
    Return:
    json file of the new status
    """
    for value in controls:
        controls[value] = False
    controls[direction] = not controls[direction]
    return jsonify({direction: controls[direction]})

@app.route("/stop", methods=["POST"])
def stop():
    """
    Turns all movement to False
    
    Parameters:
    None
    
    Return:
    None
    """
    for value in controls:
        controls[value] = False


@app.route("/status", methods=["GET"])
def status():
    """
    Just displays all the controls to the /status branch
    
    Parameters:
    None
    
    Return:
    The new status of the controls
    """
    return controls

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port=5000)
