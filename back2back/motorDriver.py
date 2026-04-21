import cv2
import requests
import base64
import threading
from PCA9685 import PCA9685
import time

MOTOR_SPEED = 35
TURN_SPEED = 52

API_STATUS_URL = "http://192.168.240.5:5000/status"
API_UPLOAD_URL = "http://192.168.240.5:5000/upload_frame"


def send_frame(frame_b64):
    try:
        requests.post(API_UPLOAD_URL, json={"frame": frame_b64}, timeout=1)
    except Exception as e:
        print("Error sending frame:", e)


cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

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
        if speed > 100:
            return

        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, speed)

        if motor_id == 0:
            if direction == 'backward':
                pwm.setLevel(self.AIN1, 0)
                pwm.setLevel(self.AIN2, 1)
            else:
                pwm.setLevel(self.AIN1, 1)
                pwm.setLevel(self.AIN2, 0)
        else:
            if direction == 'backward':
                pwm.setLevel(self.BIN1, 1)
                pwm.setLevel(self.BIN2, 0)
            else:
                pwm.setLevel(self.BIN1, 0)
                pwm.setLevel(self.BIN2, 1)

    def MotorStop(self, motor_id):
        pwm.setDutycycle(self.PWMA if motor_id == 0 else self.PWMB, 0)


Motor = MotorDriver()


def execute_command(command):
    print(f"Executing command: {command}")

    if command == 'backward':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)

    elif command == 'forward':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)

    elif command == 'left':
        Motor.MotorRun(0, 'backward', max(0, MOTOR_SPEED - 8))
        Motor.MotorRun(1, 'backward', min(100, MOTOR_SPEED + 8))

    elif command == 'right':
        Motor.MotorRun(0, 'backward', min(100, MOTOR_SPEED + 8))
        Motor.MotorRun(1, 'backward', max(0, MOTOR_SPEED - 8))

    elif command == 'turn_left_hard':
        Motor.MotorRun(0, 'forward', TURN_SPEED)
        Motor.MotorRun(1, 'backward', TURN_SPEED)

    elif command == 'turn_right_hard':
        Motor.MotorRun(0, 'backward', TURN_SPEED)
        Motor.MotorRun(1, 'forward', TURN_SPEED)

    elif command == 'stop':
        Motor.MotorStop(0)
        Motor.MotorStop(1)

    else:
        Motor.MotorStop(0)
        Motor.MotorStop(1)


current_state = 'stop'

print("Raspberry Pi robot client started...")

while True:
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        threading.Thread(target=send_frame, args=(frame_b64,), daemon=True).start()

    try:
        response = requests.get(API_STATUS_URL, timeout=1)
        response.raise_for_status()
        data = response.json()

        if data.get("autonomous", False):
            new_state = data.get("auto_command", "stop")
        else:
            new_state = data.get("manual_command", "stop")

        if new_state != current_state:
            execute_command(new_state)
            current_state = new_state

    except requests.exceptions.ConnectionError:
        if current_state != 'error':
            print("WARNING: Could not connect to API. Stopping motors.")
            execute_command('stop')
            current_state = 'error'

    except requests.exceptions.Timeout:
        if current_state != 'timeout':
            print("WARNING: API request timed out. Stopping motors.")
            execute_command('stop')
            current_state = 'timeout'

    except Exception as e:
        print(f"Unexpected error: {e}")
        execute_command('stop')
        current_state = 'error'

    time.sleep(0.03)
