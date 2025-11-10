# This file is now the client running on the Raspberry Pi.

# It polls the Mac API server (172.20.140.4:5000) for the current command state.


import cv2
import requests
import base64
import threading
from PCA9685 import PCA9685
import time




# --- Configuration ---

# Fix: Added comma to Dir list and defined all possible states

Dir = [

    'forward',

    'backward',

    'right',

    'left'	

]



# Speed setting for the motors (as per original code)

MOTOR_SPEED = 35



# The URL of the Flask server (api.py) running on your Mac

# This IP must be accessible from the Raspberry Pi.

API_STATUS_URL = "http://192.168.240.9:5000/status" 


API_URL = "http://192.168.240.9:5000/upload_frame"

def send_frame(frame_b64):
    try:
        requests.post(API_URL, json={"frame": frame_b64}, timeout=1)
    except Exception as e:
        print("Error sending frame:", e)

cap = cv2.VideoCapture(0)

# optional: make it smaller for speed
cap.set(3, 320)  # width
cap.set(4, 240)  # height

# Initialize PWM Driver

pwm = PCA9685(0x40, debug=False)

pwm.setPWMFreq(50)



# --- Motor Driver Class (Adjusted for Clarity) ---

class MotorDriver():

    def __init__(self):

        # Define the PWM/Driver pins

        self.PWMA = 0 # Left Motor Speed PWM Channel

        self.AIN1 = 1 # Left Motor Direction 1

        self.AIN2 = 2 # Left Motor Direction 2


        self.PWMB = 5 # Right Motor Speed PWM Channel

        self.BIN1 = 3 # Right Motor Direction 1

        self.BIN2 = 4 # Right Motor Direction 2



    def MotorRun(self, motor_id, index, speed):

        """

        Runs a single motor in a specified direction.

        motor_id: 0 for Left, 1 for Right

        index: 'forward' or 'backward'

        """

        if speed > 100:

            return



        # Set PWM Duty Cycle (Speed)

        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)



        if motor_id == 0: # Left Motor

            if index == 'backward':

                print("Left Motor: Forward")

                pwm.setLevel(self.AIN1, 0) # Low

                pwm.setLevel(self.AIN2, 1) # High

            else: # Must be 'backward'

                print("Left Motor: Backward")

                pwm.setLevel(self.AIN1, 1) # High

                pwm.setLevel(self.AIN2, 0) # Low

        else: # Right Motor (motor_id == 1)

            if index == 'backward':

                print("Right Motor: Forward")

                pwm.setLevel(self.BIN1, 1) # High

                pwm.setLevel(self.BIN2, 0) # Low

            else: # Must be 'backward'

                print("Right Motor: Backward")

                pwm.setLevel(self.BIN1, 0) # Low

                pwm.setLevel(self.BIN2, 1) # High
        print("True")


    def MotorStop(self, motor_id):

        """Stops a single motor by setting its PWM duty cycle to 0."""

        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)



# Initialize the motor driver

Motor = MotorDriver()





# --- Main Control Loop (Polling the Mac API) ---

def execute_command(command):
    print(f"Executing new command: {command}")

    """Executes the motor actions based on the command received from the API."""

    

    # 0 = Left Motor, 1 = Right Motor (Assuming standard differential drive)

    

    if command == 'backward':

        Motor.MotorRun(0, 'forward', MOTOR_SPEED)

        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
        print("motor working")

    elif command == 'forward':

        Motor.MotorRun(0, 'backward', MOTOR_SPEED)

        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
        print("back motor working")
    elif command == 'left':

        # Pivot Left: Left motor backward, Right motor forward

        Motor.MotorRun(0, 'backward', MOTOR_SPEED)

        Motor.MotorRun(1, 'forward', MOTOR_SPEED)

    elif command == 'right':

        # Pivot Right: Left motor forward, Right motor backward

        Motor.MotorRun(0, 'forward', MOTOR_SPEED)

        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
    elif command == 'stop':
        Motor.MotorStop(0)
        Motor.MotorStop(1)
        print("Stopping motors")
    else:


        Motor.MotorStop(0)

        Motor.MotorStop(1)
        print("Stopping motors error thing")




current_state = 'stop'



print("Raspberry Pi Motor Client started. Polling API server...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # compress the frame (lower = faster)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # send it in background thread
    threading.Thread(target=send_frame, args=(frame_b64,)).start()
    try:

        # 1. Poll the Mac API for the current command

        response = requests.get(API_STATUS_URL, timeout=1)

        response.raise_for_status()

        # 2. Extract the current direction

        #new_state = response.json().get('direction', 'stop')
	# 2. Extract the current direction

        data=response.json()

        #new_state = response.json().get('direction', 'stop')

        print(f"JSON Data: {data}")

        new_state = "stop"

        for direction in data:
                if data[direction] == True:

                        new_state = direction

        # 3. Only execute motor commands if the state has changed

        if new_state != current_state:

            print(f"Executing new command: {new_state}")

            execute_command(new_state)

            current_state = new_state



    except requests.exceptions.ConnectionError:

        # Handle case where the Mac server is not running or unreachable

        if current_state != 'error':

            print(f"WARNING: Could not connect to API at {API_STATUS_URL}. Stopping motors.")

            execute_command('stop')

            current_state = 'error' # Prevent repeated warnings

    except requests.exceptions.Timeout:

        # Handle case where the API is slow to respond

        if current_state != 'timeout':

            print("WARNING: API request timed out. Stopping motors.")

            execute_command('stop')

            current_state = 'timeout'

    except Exception as e:

        # General error handling

        print(f"An unexpected error occurred: {e}. Stopping motors.")

        execute_command('stop')

        current_state = 'error'



    # Wait for a short period before polling again (polling rate)


    time.sleep(0.03)
