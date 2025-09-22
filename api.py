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
    if direction in controls:
        controls[direction] = not controls[direction]
        return jsonify({direction: controls[direction]})

@app.route("/status", methods=["GET"])
def status():
    return jsonify(controls)

if __name__ == "__main__":
    app.run(port=5000)