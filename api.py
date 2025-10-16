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
    for value in controls:
        controls[value] = False
    controls[direction] = not controls[direction]
    return jsonify({direction: controls[direction]})

@app.route("/stop", methods=["POST"])
def stop():
    for value in controls:
        controls[value] = False


@app.route("/status", methods=["GET"])
def status():
    #return jsonify(controls)
    return controls

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port=5000)