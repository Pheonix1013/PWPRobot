#import flask and jsonify function from library
from flask import Flask, jsonify

#create API
app = Flask(__name__)

# robot controls state
controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}

#define the branch of the api and the method "post"
@app.route("/<direction>", methods=["POST"])

#create move function
def move(direction):
    """
    Updates the direction in the dictionary
    If value for the specific direction is True, it well set to False and vice versa.
    
    Parameters:
    Direction
    
    Return:
    Jsonified dictionary with updated controls
    """
    
    if direction in controls:
        controls[direction] = not controls[direction]
        return jsonify({direction: controls[direction]})

#define the branch of the api and the method "get"
@app.route("/status", methods=["GET"])

#create the status function
def status():
    """
    Displays the current state of the controls
    
    Parameters:
    None
    
    Return:
    Jsonified list of current controls
	"""
    
    return jsonify(controls)


if __name__ == "__main__":
    app.run(port=5000)