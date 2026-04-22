import cv2
import requests
import base64
import threading
from PCA9685 import PCA9685
import time

MOTOR_SPEED = 15
TURN_INTENSITY = 12 # Adjust this to make turns sharper/gentler

API_STATUS_URL = "http://192.168.240.2:5000/status"
API_UPLOAD_URL = "http://192.168.240.2:5000/upload_frame"

def send_frame(frame_b64):
    try:
        requests.post(API_UPLOAD_URL, json={"frame": frame_b64}, timeout=1)
    except:
        pass

cap = cv2.VideoCapture(0)
cap.set(3, 320); cap.set(4, 240)
pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

class MotorDriver():
    def __init__(self):
        self.PWMA, self.AIN1, self.AIN2 = 0, 1, 2
        self.PWMB, self.BIN1, self.BIN2 = 5, 3, 4

    def MotorRun(self, motor_id, direction, speed):
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)
        if motor_id == 0:
            pwm.setLevel(self.AIN1, 1 if direction == 'forward' else 0)
            pwm.setLevel(self.AIN2, 0 if direction == 'forward' else 1)
        else:
            pwm.setLevel(self.BIN1, 0 if direction == 'forward' else 1)
            pwm.setLevel(self.BIN2, 1 if direction == 'forward' else 0)

    def MotorStop(self, motor_id):
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)

Motor = MotorDriver()

def execute_command(command):
    if command == 'backward':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
    elif command == 'forward':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
    elif command == 'right':
        Motor.MotorRun(0, 'forward', max(0, MOTOR_SPEED - TURN_INTENSITY))
        Motor.MotorRun(1, 'forward', min(100, MOTOR_SPEED + TURN_INTENSITY))
    elif command == 'left':
        Motor.MotorRun(0, 'forward', min(100, MOTOR_SPEED + TURN_INTENSITY))
        Motor.MotorRun(1, 'forward', max(0, MOTOR_SPEED - TURN_INTENSITY))
    else:
        Motor.MotorStop(0); Motor.MotorStop(1)

current_state = 'stop'
while True:
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        f_b64 = base64.b64encode(buffer).decode('utf-8')
        threading.Thread(target=send_frame, args=(f_b64,), daemon=True).start()

    try:
        data = requests.get(API_STATUS_URL, timeout=1).json()
        new_state = data.get("auto_command") if data.get("autonomous") else data.get("manual_command")
        if new_state != current_state:
            execute_command(new_state)
            current_state = new_state
    except:
        execute_command('stop')
    time.sleep(0.03)
