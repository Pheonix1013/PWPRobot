#import necessary functions
import cv2
import requests
import base64
import threading
from PCA9685 import PCA9685
import time

# --- Configuration ---

Dir = [
    'forward',
    'backward',
    'right',
    'left'
]

MOTOR_SPEED = 35

API_STATUS_URL = "http://192.168.240.9:5000/status"
API_URL = "http://192.168.240.9:5000/upload_frame"


latest_to_send = None
send_lock = threading.Lock()

def sender_loop():
    global latest_to_send
    while True:
        payload = None
        with send_lock:
            if latest_to_send is not None:
                payload = latest_to_send
                latest_to_send = None

        if payload is not None:
            try:
                requests.post(API_URL, json={"frame": payload}, timeout=1)
            except Exception as e:
                print("Error sending frame:", e)

        time.sleep(0.01)

threading.Thread(target=sender_loop, daemon=True).start()

cap = cv2.VideoCapture(0)

cap.set(3, 320)  # width
cap.set(4, 240)  # height

# Initialize PWM Driver
pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

# motors

class MotorDriver():

    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2

        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

    def MotorRun(self, motor_id, index, speed):
        if speed > 100:
            return

        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)

        if motor_id == 0:  # Left Motor
            if index == 'backward':
                print("Left Motor: Forward")
                pwm.setLevel(self.AIN1, 0)
                pwm.setLevel(self.AIN2, 1)
            else:
                print("Left Motor: Backward")
                pwm.setLevel(self.AIN1, 1)
                pwm.setLevel(self.AIN2, 0)
        else:  # Right Motor
            if index == 'backward':
                print("Right Motor: Forward")
                pwm.setLevel(self.BIN1, 1)
                pwm.setLevel(self.BIN2, 0)
            else:
                print("Right Motor: Backward")
                pwm.setLevel(self.BIN1, 0)
                pwm.setLevel(self.BIN2, 1)

        print("True")

    def MotorStop(self, motor_id):
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)

Motor = MotorDriver()

#main loop

def execute_command(command):
    print(f"Executing new command: {command}")

    if command == 'backward':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
        print("motor working")

    elif command == 'forward':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
        print("back motor working")

    elif command == 'left':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)

    elif command == 'right':
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

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    with send_lock:
        latest_to_send = frame_b64

    try:
        response = requests.get(API_STATUS_URL, timeout=1)
        response.raise_for_status()

        data = response.json()
        print(f"JSON Data: {data}")

        new_state = "stop"
        for direction in data:
            if data[direction] == True:
                new_state = direction

        if new_state != current_state:
            print(f"Executing new command: {new_state}")
            execute_command(new_state)
            current_state = new_state

    except requests.exceptions.ConnectionError:
        if current_state != 'error':
            print(f"WARNING: Could not connect to API at {API_STATUS_URL}. Stopping motors.")
            execute_command('stop')
            current_state = 'error'

    except requests.exceptions.Timeout:
        if current_state != 'timeout':
            print("WARNING: API request timed out. Stopping motors.")
            execute_command('stop')
            current_state = 'timeout'

    except Exception as e:
        print(f"An unexpected error occurred: {e}. Stopping motors.")
        execute_command('stop')
        current_state = 'error'

    time.sleep(0.03)
