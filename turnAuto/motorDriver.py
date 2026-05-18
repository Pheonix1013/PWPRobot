import cv2
import requests
import base64
import queue
import threading
import time
from PCA9685 import PCA9685

MOTOR_SPEED = 13
TURN_INTENSITY = 12

# Proportional steering for autonomous mode.  The API sends a signed pixel
# error: negative means the tape center is left of camera center, positive is
# right.  auto_speed lets the API briefly stop after losing the tape, then
# search slowly toward the last known direction.
KP = 0.12       # gain: error (pixels) -> speed differential
MAX_STEER = 10  # max speed differential added to either motor
PAPER_TURN_SPEED = min(100, MOTOR_SPEED + TURN_INTENSITY + 8)
MIN_AUTO_SPEED = 1

API_STATUS_URL = "http://192.168.240.2:5000/status"
API_UPLOAD_URL = "http://192.168.240.2:5000/upload_frame"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

# Single sender thread with a 1-frame queue — drops frames if sender is busy
frame_queue = queue.Queue(maxsize=1)

def frame_sender():
    while True:
        frame_b64 = frame_queue.get()
        try:
            requests.post(API_UPLOAD_URL, json={"frame": frame_b64}, timeout=1)
        except:
            pass

threading.Thread(target=frame_sender, daemon=True).start()

def execute_command(command):
    if command == 'backward':
        Motor.MotorRun(0, 'forward', MOTOR_SPEED)
        Motor.MotorRun(1, 'forward', MOTOR_SPEED)
    elif command == 'forward':
        Motor.MotorRun(0, 'backward', MOTOR_SPEED)
        Motor.MotorRun(1, 'backward', MOTOR_SPEED)
    elif command == 'left':
        Motor.MotorRun(0, 'forward', max(0, MOTOR_SPEED - TURN_INTENSITY))
        Motor.MotorRun(1, 'forward', min(100, MOTOR_SPEED + TURN_INTENSITY))
    elif command == 'right':
        Motor.MotorRun(0, 'forward', min(100, MOTOR_SPEED + TURN_INTENSITY))
        Motor.MotorRun(1, 'forward', max(0, MOTOR_SPEED - TURN_INTENSITY + 2))
    else:
        Motor.MotorStop(0); Motor.MotorStop(1)

current_state = 'stop'
while True:
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        f_b64 = base64.b64encode(buffer).decode('utf-8')
        try:
            frame_queue.put_nowait(f_b64)
        except queue.Full:
            pass  # sender still busy, drop this frame

    try:
        data = requests.get(API_STATUS_URL, timeout=1).json()

        if data.get("martian_detected") or data.get("robot_state") in ("MARTIAN_DETECTED", "MARTIAN_STOPPED"):
            Motor.MotorStop(0)
            Motor.MotorStop(1)
            current_state = "martian_stop"
        elif data.get("autonomous") or data.get("robot_state") == "PAPER_AVOIDANCE":
            auto_cmd = data.get("auto_command", "stop")
            error = data.get("auto_error", 0)

            speed_scale = float(data.get("auto_speed", 1.0))
            base_speed = int(MOTOR_SPEED * max(0.0, min(1.0, speed_scale)))

            avoidance_phase = data.get("avoidance_phase", "idle")

            if auto_cmd == "stop" or base_speed < MIN_AUTO_SPEED:
                Motor.MotorStop(0)
                Motor.MotorStop(1)
            elif auto_cmd == "backward":
                Motor.MotorRun(0, 'forward', max(MIN_AUTO_SPEED, base_speed))
                Motor.MotorRun(1, 'forward', max(MIN_AUTO_SPEED, base_speed))
            elif data.get("robot_state") == "PAPER_AVOIDANCE" and avoidance_phase == "turn_away" and auto_cmd in ("left", "right"):
                # Hard pivot only during the dedicated obstacle-avoidance turn phase;
                # normal autonomous steering below remains a smooth proportional arc.
                if auto_cmd == "right":
                    Motor.MotorRun(0, 'backward', PAPER_TURN_SPEED)
                    Motor.MotorRun(1, 'forward', PAPER_TURN_SPEED)
                else:
                    Motor.MotorRun(0, 'forward', PAPER_TURN_SPEED)
                    Motor.MotorRun(1, 'backward', PAPER_TURN_SPEED)
            else:
                # Proportional steering: scale error into a speed differential.
                # Positive error = tape center is right of frame center = steer right.
                # motor 0 (left side) faster turns robot right, motor 1 (right side) faster turns left.
                steer = int(KP * error)
                steer = max(-MAX_STEER, min(MAX_STEER, steer))
                Motor.MotorRun(0, 'backward', max(MIN_AUTO_SPEED, base_speed + steer))
                Motor.MotorRun(1, 'backward', max(MIN_AUTO_SPEED, base_speed - steer))

            current_state = f"auto:{auto_cmd}"
        else:
            new_state = data.get("manual_command", "stop")
            if new_state != current_state:
                execute_command(new_state)
                current_state = new_state

    except:
        Motor.MotorStop(0)
        Motor.MotorStop(1)

    time.sleep(0.03)
