import cv2
import requests
import base64
import time
import threading
from PCA9685 import PCA9685

# --- Configuration ---
# Replace with the actual IP of the computer running api.py
SERVER_IP = "192.168.240.2" 
API_STATUS_URL = f"http://{SERVER_IP}:5000/status"
API_UPLOAD_URL = f"http://{SERVER_IP}:5000/upload_frame"

MOTOR_SPEED = 35
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# --- Initialize Hardware ---
pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

class MotorDriver():
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

    def MotorRun(self, motor_id, direction, speed):
        speed = max(0, min(100, speed))
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)

        if motor_id == 0: # Left Motor
            if direction == 'forward':
                pwm.setLevel(self.AIN1, 1)
                pwm.setLevel(self.AIN2, 0)
            else:
                pwm.setLevel(self.AIN1, 0)
                pwm.setLevel(self.AIN2, 1)
        else: # Right Motor
            if direction == 'forward':
                pwm.setLevel(self.BIN1, 0)
                pwm.setLevel(self.BIN2, 1)
            else:
                pwm.setLevel(self.BIN1, 1)
                pwm.setLevel(self.BIN2, 0)

    def MotorStop(self, motor_id):
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)

Motor = MotorDriver()

# --- Robot Logic ---

def execute_command(command):
    """Maps string commands to motor movements"""
    if command == 'forward':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
    elif command == 'backward':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
    elif command == 'left':
        # Slight pivot or differential turn
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
    elif command == 'right':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
    else: # 'stop'
        Motor.MotorStop(0)
        Motor.MotorStop(1)

def send_frame_async(frame_b64):
    """Sends frame to server without blocking the main loop"""
    try:
        requests.post(API_UPLOAD_URL, json={"frame": frame_b64}, timeout=0.2)
    except:
        pass

# --- Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

current_state = 'stop'
print(f"Robot Client Online. Connecting to {SERVER_IP}...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. Encode and upload frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        threading.Thread(target=send_frame_async, args=(frame_b64,), daemon=True).start()

        # 2. Get latest command from Server
        try:
            response = requests.get(API_STATUS_URL, timeout=0.2)
            data = response.json()

            # Prioritize autonomous if enabled
            if data.get("autonomous", False):
                new_state = data.get("auto_command", "stop")
            else:
                new_state = data.get("manual_command", "stop")

            # 3. Apply movement
            if new_state != current_state:
                print(f"Switching to: {new_state}")
                execute_command(new_state)
                current_state = new_state

        except Exception as e:
            if current_state != 'stop':
                execute_command('stop')
                current_state = 'stop'
            print("Connection to API lost...")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Shutting down...")
    Motor.MotorStop(0)
    Motor.MotorStop(1)
    cap.release()
