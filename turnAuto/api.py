from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import base64
import cv2
import numpy as np
import os
from pathlib import Path
import threading

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

app = FastAPI()

# --- GLOBALS ---
latest_frame = None
latest_frame_raw = None
autonomous_enabled = False
auto_command = "stop"
auto_error = 0
lane_status = "No lanes"
last_steering_cmd = "stop"
smoothed_error = 0
lost_frame_count = 0
auto_speed_scale = 1.0
martian_detected = False
martian_confidence = None
martian_bbox = None
martian_stop_latched = False
robot_state = "STOPPED"
paper_ball_detected = False
paper_ball_confidence = None
paper_ball_bbox = None
avoidance_phase = "idle"
avoidance_counter = 0
paper_cooldown_frames = 0
paper_clear_frames = 0
paper_rearm_required = False
avoidance_turn_error = 90

# Thread lock for globals written from executor + read from streaming endpoints
_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controls = {
    "forward": False,
    "backward": False,
    "left": False,
    "right": False
}

# --- Martian SIFT/template detector ---
# Kept inside this TurnAuto API file so TurnAuto does not import detector code
# from another folder. Place marvin_template.png in this folder to enable detection.
MARTIAN_TEMPLATE_PATH = Path(__file__).resolve().parent / "marvin_template.png"
MARTIAN_CONFIDENCE_THRESHOLD = 0.60
MARTIAN_MIN_FRAME_DESCRIPTORS = 20
MARTIAN_MIN_GOOD_MATCHES = 20
MARTIAN_MIN_INLIERS = 15


class MartianDetector:
    def __init__(self, template_path=MARTIAN_TEMPLATE_PATH, confidence_threshold=MARTIAN_CONFIDENCE_THRESHOLD):
        self.template_path = Path(template_path)
        self.confidence_threshold = confidence_threshold
        self.available = False
        self._template = None
        self._template_keypoints = None
        self._template_descriptors = None
        sift_factory = getattr(cv2, "SIFT_create", None)
        if sift_factory is None:
            print("Martian detector disabled: this OpenCV build does not provide cv2.SIFT_create")
            self._sift = None
            self._flann = None
            return
        self._sift = sift_factory()
        self._flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self._load_template()

    def _load_template(self):
        if not self.template_path.exists():
            print(f"Martian detector disabled: missing template file {self.template_path}")
            return

        self._template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        if self._template is None:
            print(f"Martian detector disabled: could not read template file {self.template_path}")
            return

        self._template_keypoints, self._template_descriptors = self._sift.detectAndCompute(self._template, None)
        if self._template_descriptors is None or len(self._template_descriptors) < MARTIAN_MIN_GOOD_MATCHES:
            print(f"Martian detector disabled: template has too few SIFT features {self.template_path}")
            return

        self.available = True

    def detect(self, frame):
        result = {"detected": False, "confidence": None, "bbox": None}
        if not self.available:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self._sift.detectAndCompute(gray, None)
        if frame_descriptors is None or len(frame_descriptors) < MARTIAN_MIN_FRAME_DESCRIPTORS:
            return result

        matches = self._flann.knnMatch(self._template_descriptors, frame_descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        if len(good) <= MARTIAN_MIN_GOOD_MATCHES:
            return result

        src_pts = np.float32([self._template_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if transform is None or mask is None:
            return result

        inliers = int(mask.ravel().sum())
        confidence = min(1.0, inliers / 25.0)
        if inliers <= MARTIAN_MIN_INLIERS or confidence < self.confidence_threshold:
            result["confidence"] = confidence
            return result

        h, w = self._template.shape
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, transform)
        x, y, bw, bh = cv2.boundingRect(np.int32(projected))
        return {
            "detected": True,
            "confidence": confidence,
            "bbox": [int(x), int(y), int(bw), int(bh)],
        }


# Thresholds
# HSV is used only to collect candidate colored pixels.  A candidate does not
# become a path until the component/contour is proven thick enough to be tape.
TAPE_HSV_RANGES = (
    (np.array([35, 45, 45]), np.array([90, 255, 255])),     # green
    (np.array([18, 55, 60]), np.array([38, 255, 255])),     # yellow
    (np.array([90, 55, 45]), np.array([140, 255, 255])),    # blue
    (np.array([5, 70, 60]), np.array([24, 255, 255])),      # orange
    (np.array([125, 45, 45]), np.array([165, 255, 255])),   # purple
)
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([140, 255, 255])
HORIZONTAL_MIN_WIDTH_RATIO = 0.45
HORIZONTAL_MAX_HEIGHT = 90
BOTTOM_STOP_ZONE_RATIO = 0.82

ROI_TOP_RATIO = 0.42
MIN_TAPE_AREA = 850
MIN_TAPE_THICKNESS_PX = 14
MIN_TAPE_LENGTH_PX = 35
MIN_TAPE_FILL_RATIO = 0.16
CENTER_DEADBAND = 32
SEARCH_ERROR = 90
LOST_STOP_FRAMES = 4
ERROR_SMOOTHING = 0.35
PARTIAL_EDGE_MARGIN = 8
# Paper-ball detection mirrors obstacleDetection/main.py: find objects that are
# both moving according to background subtraction and white in HSV.
PAPER_WHITE_LOWER = np.array([0, 0, 180])
PAPER_WHITE_UPPER = np.array([180, 50, 255])
PAPER_MIN_AREA = 500
PAPER_DETECT_ROI_TOP_RATIO = 0.60
PAPER_DETECT_ROI_BOTTOM_RATIO = 0.95
PAPER_BACKGROUND_HISTORY = 100
PAPER_BACKGROUND_VAR_THRESHOLD = 50
PAPER_BACKUP_FRAMES = 15
PAPER_TURN_FRAMES = 36
PAPER_CLEAR_FORWARD_FRAMES = 26
PAPER_RECOVER_FRAMES = 22
PAPER_TURN_ERROR = 180
PAPER_COOLDOWN_FRAMES = 25
PAPER_REARM_CLEAR_FRAMES = 10

# --- PERF: kernels created once at module level, not per-frame ---
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_BLUE_KERNEL = np.ones((5, 5), np.uint8)
_TAPE_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_TAPE_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_PAPER_KERNEL = np.ones((5, 5), np.uint8)
_PAPER_BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(
    history=PAPER_BACKGROUND_HISTORY,
    varThreshold=PAPER_BACKGROUND_VAR_THRESHOLD,
    detectShadows=True,
)

martian_detector = MartianDetector(confidence_threshold=MARTIAN_CONFIDENCE_THRESHOLD)


@app.post("/stop")
async def stop():
    global autonomous_enabled, auto_command, auto_speed_scale, martian_stop_latched, robot_state, avoidance_phase, avoidance_counter, paper_rearm_required, paper_clear_frames
    autonomous_enabled = False
    auto_command = "stop"
    auto_speed_scale = 0.0
    martian_stop_latched = False
    avoidance_phase = "idle"
    avoidance_counter = 0
    paper_rearm_required = False
    paper_clear_frames = 0
    robot_state = "STOPPED"
    for k in controls:
        controls[k] = False
    return {"message": "All movements stopped", "autonomous": autonomous_enabled}

@app.post("/autonomous/start")
async def autonomous_start():
    global autonomous_enabled, auto_command, auto_speed_scale, lost_frame_count, smoothed_error, last_steering_cmd, martian_stop_latched, robot_state, avoidance_phase, avoidance_counter, paper_rearm_required, paper_clear_frames
    for k in controls:
        controls[k] = False
    autonomous_enabled = True
    martian_stop_latched = False
    robot_state = "AUTONOMOUS"
    auto_command = "stop"
    auto_speed_scale = 0.0
    lost_frame_count = 0
    smoothed_error = 0
    last_steering_cmd = "stop"
    avoidance_phase = "idle"
    avoidance_counter = 0
    paper_rearm_required = False
    paper_clear_frames = 0
    return {"autonomous": True}

@app.post("/autonomous/stop")
async def autonomous_stop():
    global autonomous_enabled, auto_command, auto_speed_scale, martian_stop_latched, robot_state, avoidance_phase, avoidance_counter, paper_rearm_required, paper_clear_frames
    autonomous_enabled = False
    auto_command = "stop"
    auto_speed_scale = 0.0
    martian_stop_latched = False
    avoidance_phase = "idle"
    avoidance_counter = 0
    paper_rearm_required = False
    paper_clear_frames = 0
    robot_state = "STOPPED"
    return {"autonomous": False}


@app.post("/martian/reset")
async def martian_reset():
    global autonomous_enabled, auto_command, auto_error, auto_speed_scale, martian_stop_latched, robot_state, lane_status, avoidance_phase, avoidance_counter, paper_rearm_required, paper_clear_frames
    autonomous_enabled = False
    auto_command = "stop"
    auto_error = 0
    auto_speed_scale = 0.0
    martian_stop_latched = False
    avoidance_phase = "idle"
    avoidance_counter = 0
    paper_rearm_required = False
    paper_clear_frames = 0
    robot_state = "STOPPED"
    lane_status = "Martian stop reset | stopped"
    for k in controls:
        controls[k] = False
    return {"robot_state": robot_state, "martian_latched": martian_stop_latched}


def build_colored_tape_candidate_mask(hsv, h, w):
    """Return HSV candidate pixels for the supported tape colors inside the floor ROI."""
    color_mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in TAPE_HSV_RANGES:
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

    # Ignore the upper image where walls/robot parts are more likely than floor tape.
    roi_mask = np.zeros_like(color_mask)
    top_y = int(h * ROI_TOP_RATIO)
    poly = np.array([
        [int(w * 0.01), h - 1],
        [int(w * 0.99), h - 1],
        [int(w * 0.74), top_y],
        [int(w * 0.26), top_y],
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [poly], 255)
    color_mask = cv2.bitwise_and(color_mask, roi_mask)

    # Clean camera speckles but do not inflate a grout line enough to pass thickness validation.
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, _TAPE_OPEN_KERNEL, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, _TAPE_CLOSE_KERNEL, iterations=1)
    return color_mask


def filter_tape_by_thickness(candidate_mask):
    """Keep only colored components that are physically thick enough to be floor tape."""
    valid_mask = np.zeros_like(candidate_mask)
    detections = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < MIN_TAPE_AREA:
            continue
        if max(bw, bh) < MIN_TAPE_LENGTH_PX:
            continue

        fill_ratio = area / float(max(1, bw * bh))
        if fill_ratio < MIN_TAPE_FILL_RATIO:
            continue

        component = (labels[y:y + bh, x:x + bw] == label).astype(np.uint8) * 255

        # Distance transform measures the largest inscribed radius.  Doubling it gives
        # an orientation-independent tape thickness estimate, so thin grout/edge lines
        # are rejected even if their color/contrast looks line-like.
        radius = float(cv2.distanceTransform(component, cv2.DIST_L2, 5).max())
        thickness = radius * 2.0
        if thickness < MIN_TAPE_THICKNESS_PX:
            continue

        valid_mask[labels == label] = 255
        detections.append({
            "label": label,
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "area": int(area),
            "thickness": thickness,
        })

    return valid_mask, detections


def estimate_path_from_tape(valid_mask, detections, w, h):
    """Estimate the path center from the thick tape blob nearest the robot."""
    if not detections:
        return None

    # Prefer a thick component that reaches low in the frame because that is what the
    # robot will drive over next.
    best = max(detections, key=lambda d: (d["bbox"][1] + d["bbox"][3]) * 3 + d["area"] / 500.0)
    x, y, bw, bh = best["bbox"]
    component = np.zeros_like(valid_mask)
    component[y:y + bh, x:x + bw] = valid_mask[y:y + bh, x:x + bw]

    y1 = max(y, int(h * 0.62))
    y2 = min(y + bh, int(h * 0.94))
    if y2 <= y1:
        y1, y2 = y, y + bh

    centers = []
    widths = []
    for row_y in range(y1, y2):
        xs = np.flatnonzero(component[row_y])
        if xs.size == 0:
            continue
        centers.append(float((xs[0] + xs[-1]) / 2.0))
        widths.append(float(xs[-1] - xs[0] + 1))

    if not centers:
        return None

    visible_center = float(np.average(centers, weights=np.maximum(widths, 1.0)))
    visible_width = float(np.median(widths))
    half_tape = max(best["thickness"] / 2.0, MIN_TAPE_THICKNESS_PX / 2.0)

    touches_left = x <= PARTIAL_EDGE_MARGIN
    touches_right = x + bw >= w - PARTIAL_EDGE_MARGIN
    partial = False
    if touches_left and not touches_right:
        # Only the right side of a tape that continues off the left edge is visible.
        path_center = max(0.0, x + bw - half_tape)
        partial = True
    elif touches_right and not touches_left:
        # Only the left side of a tape that continues off the right edge is visible.
        path_center = min(float(w - 1), x + half_tape)
        partial = True
    else:
        path_center = visible_center

    return {
        "center": int(max(0, min(w - 1, round(path_center)))),
        "visible_center": int(max(0, min(w - 1, round(visible_center)))),
        "width": visible_width,
        "bbox": best["bbox"],
        "thickness": best["thickness"],
        "partial": partial,
    }


def detect_blue_stop_line(frame, hsv):
    # Stop-line detection is intentionally separate from path following.  It still
    # requires a wide, thick blue horizontal blob so ordinary blue path tape does not
    # stop the robot unless it spans the lower frame like a finish/stop marker.
    h, w = frame.shape[:2]
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, _BLUE_KERNEL)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, _BLUE_KERNEL)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_detected = False
    best_line = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 700:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        component = np.zeros((bh, bw), dtype=np.uint8)
        shifted = cnt - np.array([[[x, y]]])
        cv2.drawContours(component, [shifted], -1, 255, -1)
        thickness = cv2.distanceTransform(component, cv2.DIST_L2, 5).max() * 2.0
        width_ratio = bw / float(w)
        bottom_y = y + bh
        if (
            thickness >= MIN_TAPE_THICKNESS_PX
            and bw > bh * 2.5
            and width_ratio >= HORIZONTAL_MIN_WIDTH_RATIO
            and bh <= HORIZONTAL_MAX_HEIGHT
        ):
            if area > best_area:
                best_area = area
                line_y = y + bh // 2
                best_line = (x, line_y, x + bw, line_y, bottom_y >= int(h * BOTTOM_STOP_ZONE_RATIO))
    if best_line:
        x1, y1, x2, y2, is_near_bottom = best_line
        stop_detected = is_near_bottom
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255) if stop_detected else (255, 0, 0), 4)
    return stop_detected, frame


def compute_auto_command(path, w):
    """Convert the validated tape position into a stable motor decision."""
    global lane_status, last_steering_cmd, auto_error, smoothed_error, lost_frame_count, auto_speed_scale
    frame_center = w // 2

    if path is not None:
        lost_frame_count = 0
        auto_speed_scale = 1.0
        raw_error = path["center"] - frame_center
        smoothed_error = int((1.0 - ERROR_SMOOTHING) * smoothed_error + ERROR_SMOOTHING * raw_error)
        if abs(raw_error) > CENTER_DEADBAND * 2:
            smoothed_error = raw_error

        if abs(smoothed_error) <= CENTER_DEADBAND:
            auto_error = 0
            last_steering_cmd = "forward"
            lane_status = f"Tape centered | thick={path['thickness']:.1f}px err={smoothed_error}"
            return "forward", path["center"], 0

        cmd = "left" if smoothed_error < 0 else "right"
        last_steering_cmd = cmd
        auto_error = smoothed_error
        mode = "partial edge" if path["partial"] else "full tape"
        lane_status = f"Tape {mode} | steer {cmd} | thick={path['thickness']:.1f}px err={smoothed_error}"
        return cmd, path["center"], smoothed_error

    lost_frame_count += 1
    auto_speed_scale = 0.0 if lost_frame_count <= LOST_STOP_FRAMES else 0.65
    if lost_frame_count <= LOST_STOP_FRAMES:
        auto_error = 0
        lane_status = "Tape lost | brief stop"
        return "stop", None, 0

    if last_steering_cmd == "left":
        auto_error = -SEARCH_ERROR
        lane_status = "Tape lost | searching left"
        return "left", None, auto_error
    if last_steering_cmd == "right":
        auto_error = SEARCH_ERROR
        lane_status = "Tape lost | searching right"
        return "right", None, auto_error

    auto_error = 0
    lane_status = "Tape lost | no previous direction"
    return "stop", None, 0

def detect_paper_ball(frame, hsv):
    """Detect a crumpled paper ball using obstacleDetection/main.py's motion+white logic."""
    h, w = frame.shape[:2]

    # 1. Motion mask from the learned floor background.
    fg_mask = _PAPER_BACKGROUND_SUBTRACTOR.apply(frame)

    # 2. Strict white mask for paper-like pixels.
    color_mask = cv2.inRange(hsv, PAPER_WHITE_LOWER, PAPER_WHITE_UPPER)

    # 3. Only keep pixels that are both moving and white, avoiding static floor glare.
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    # Match the obstacleDetection/main.py vertical gate, scaled to the live frame.
    roi = np.zeros_like(combined_mask)
    y_min = int(h * PAPER_DETECT_ROI_TOP_RATIO)
    y_max = int(h * PAPER_DETECT_ROI_BOTTOM_RATIO)
    cv2.rectangle(roi, (0, y_min), (w - 1, y_max), 255, -1)
    combined_mask = cv2.bitwise_and(combined_mask, roi)

    # 4. Same cleanup pattern as obstacleDetection/main.py.
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, _PAPER_KERNEL, iterations=1)
    combined_mask = cv2.dilate(combined_mask, _PAPER_KERNEL, iterations=3)

    # 5. Use the largest moving white object as the paper ball candidate.
    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"detected": False, "confidence": None, "bbox": None}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= PAPER_MIN_AREA:
        return {"detected": False, "confidence": None, "bbox": None}

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    if not (y_min <= center_y <= y_max):
        return {"detected": False, "confidence": None, "bbox": None}

    x, y, bw, bh = cv2.boundingRect(contour)
    confidence = float(min(1.0, area / max(PAPER_MIN_AREA * 4.0, 1.0)))
    return {
        "detected": True,
        "confidence": confidence,
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "center": [int(center_x), int(center_y)],
        "radius": float(radius),
    }


def start_paper_avoidance(result, frame_width):
    global avoidance_phase, avoidance_counter, paper_cooldown_frames, paper_rearm_required, paper_clear_frames, avoidance_turn_error, robot_state
    center = result.get("center") or [frame_width // 2, 0]
    ball_x = center[0]
    # Turn away from the side where the ball appears.  If it is centered, default
    # to the right so the maneuver is deterministic.
    avoidance_turn_error = PAPER_TURN_ERROR if ball_x <= frame_width // 2 else -PAPER_TURN_ERROR
    avoidance_phase = "backup"
    avoidance_counter = PAPER_BACKUP_FRAMES
    paper_cooldown_frames = PAPER_COOLDOWN_FRAMES
    paper_rearm_required = True
    paper_clear_frames = 0
    robot_state = "PAPER_AVOIDANCE"
    for key in controls:
        controls[key] = False


def compute_paper_avoidance_command():
    global auto_command, auto_error, auto_speed_scale, lane_status, robot_state, avoidance_phase, avoidance_counter
    robot_state = "PAPER_AVOIDANCE"

    if avoidance_phase == "backup":
        auto_command = "backward"
        auto_error = 0
        auto_speed_scale = 0.75
        lane_status = "Paper ball detected | backing up"
    elif avoidance_phase == "turn_away":
        auto_command = "right" if avoidance_turn_error > 0 else "left"
        auto_error = avoidance_turn_error
        auto_speed_scale = 0.75
        lane_status = "Paper ball avoidance | turning away"
    elif avoidance_phase == "clear_forward":
        auto_command = "right" if avoidance_turn_error > 0 else "left"
        auto_error = int(avoidance_turn_error * 0.60)
        auto_speed_scale = 0.80
        lane_status = "Paper ball avoidance | clearing obstacle"
    elif avoidance_phase == "recover_path":
        auto_command = "left" if avoidance_turn_error > 0 else "right"
        auto_error = int(-avoidance_turn_error * 0.75)
        auto_speed_scale = 0.70
        lane_status = "Paper ball avoidance | returning to path"
    else:
        return None

    avoidance_counter -= 1
    if avoidance_counter <= 0:
        if avoidance_phase == "backup":
            avoidance_phase = "turn_away"
            avoidance_counter = PAPER_TURN_FRAMES
        elif avoidance_phase == "turn_away":
            avoidance_phase = "clear_forward"
            avoidance_counter = PAPER_CLEAR_FORWARD_FRAMES
        elif avoidance_phase == "clear_forward":
            avoidance_phase = "recover_path"
            avoidance_counter = PAPER_RECOVER_FRAMES
        else:
            avoidance_phase = "idle"
            avoidance_counter = 0
    return auto_command, None, auto_error


def draw_paper_overlay(frame, result):
    bbox = result.get("bbox")
    if not bbox:
        return
    x, y, bw, bh = bbox
    center = result.get("center")
    radius = result.get("radius")
    if center and radius:
        cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 4)
    else:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 3)
    confidence = result.get("confidence")
    label = "PAPER BALL" if confidence is None else f"PAPER BALL {confidence:.2f}"
    cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def apply_martian_stop(result):
    """Latch a safe stop so the robot does not resume until the user commands it."""
    global autonomous_enabled, auto_command, auto_error, auto_speed_scale, lane_status, martian_stop_latched, robot_state
    autonomous_enabled = False
    auto_command = "stop"
    auto_error = 0
    auto_speed_scale = 0.0
    martian_stop_latched = True
    robot_state = "MARTIAN_DETECTED"
    confidence = result.get("confidence")
    if confidence is None:
        lane_status = "WE ARE NOT ALONE | stopped"
    else:
        lane_status = f"WE ARE NOT ALONE | stopped conf={confidence:.2f}"
    for key in controls:
        controls[key] = False


def draw_martian_overlay(frame, result):
    bbox = result.get("bbox")
    if not bbox:
        return
    x, y, bw, bh = bbox
    x = max(0, x)
    y = max(0, y)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
    confidence = result.get("confidence")
    label = "WE ARE NOT ALONE" if confidence is None else f"WE ARE NOT ALONE {confidence:.2f}"
    cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)


def process_frame(frame):
    global auto_command, autonomous_enabled, lane_status, auto_speed_scale, martian_detected, martian_confidence, martian_bbox, robot_state, paper_ball_detected, paper_ball_confidence, paper_ball_bbox, paper_cooldown_frames, paper_rearm_required, paper_clear_frames, avoidance_phase, avoidance_counter
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    martian_result = martian_detector.detect(frame)
    martian_detected = bool(martian_result.get("detected"))
    martian_confidence = martian_result.get("confidence")
    martian_bbox = martian_result.get("bbox")
    if martian_detected:
        apply_martian_stop(martian_result)
        draw_martian_overlay(frame, martian_result)
        return frame

    if martian_stop_latched:
        auto_command = "stop"
        auto_speed_scale = 0.0
        robot_state = "MARTIAN_STOPPED"
        lane_status = "Martian cleared | press Start Auto to resume"
        return frame

    robot_state = "AUTONOMOUS" if autonomous_enabled else "MANUAL"

    # One HSV conversion feeds the paper obstacle, stop-line, and tape detectors.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    paper_result = detect_paper_ball(frame, hsv)
    paper_ball_detected = bool(paper_result.get("detected"))
    paper_ball_confidence = paper_result.get("confidence")
    paper_ball_bbox = paper_result.get("bbox")
    if paper_ball_detected:
        draw_paper_overlay(frame, paper_result)
        paper_clear_frames = 0
        if paper_rearm_required and avoidance_phase == "idle":
            # Same ball is still visible after the scripted maneuver, so keep
            # clearing in the original away direction instead of resuming tape
            # following and driving back over it.
            avoidance_phase = "clear_forward"
            avoidance_counter = PAPER_CLEAR_FORWARD_FRAMES
        elif avoidance_phase == "recover_path":
            # Do not turn back toward the path while the obstacle is still in view.
            avoidance_phase = "clear_forward"
            avoidance_counter = PAPER_CLEAR_FORWARD_FRAMES
    elif paper_rearm_required:
        paper_clear_frames += 1
        if paper_clear_frames >= PAPER_REARM_CLEAR_FRAMES:
            paper_rearm_required = False
            paper_cooldown_frames = 0

    # Do not let a ball seen during/just after the avoidance maneuver restart the
    # maneuver.  Re-arm only after the detector has seen clear floor for several
    # frames, which prevents left/right oscillation around the same ball.
    can_start_paper_avoidance = (
        paper_ball_detected
        and avoidance_phase == "idle"
        and paper_cooldown_frames <= 0
        and not paper_rearm_required
    )
    if can_start_paper_avoidance:
        start_paper_avoidance(paper_result, w)

    avoidance_cmd = compute_paper_avoidance_command()
    if avoidance_cmd is not None:
        cv2.putText(frame, lane_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    if paper_cooldown_frames > 0:
        paper_cooldown_frames -= 1

    stop_detected, frame = detect_blue_stop_line(frame, hsv)
    candidate_mask = build_colored_tape_candidate_mask(hsv, h, w)
    valid_mask, detections = filter_tape_by_thickness(candidate_mask)
    path = estimate_path_from_tape(valid_mask, detections, w, h)
    cmd, path_center, err = compute_auto_command(path, w)

    # Debug overlay: cyan = HSV candidates, green = candidates that passed thickness.
    frame[candidate_mask > 0] = cv2.addWeighted(frame, 0.55, np.full_like(frame, (255, 255, 0)), 0.45, 0)[candidate_mask > 0]
    frame[valid_mask > 0] = cv2.addWeighted(frame, 0.45, np.full_like(frame, (0, 255, 0)), 0.55, 0)[valid_mask > 0]

    for detection in detections:
        x, y, bw, bh = detection["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{detection['thickness']:.0f}px",
            (x, max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    f_center, y_look = w // 2, int(h * 0.78)
    roi_y = int(h * ROI_TOP_RATIO)
    cv2.line(frame, (f_center, h - 1), (f_center, 0), (255, 255, 0), 2)
    cv2.line(frame, (0, y_look), (w, y_look), (255, 255, 0), 2)
    cv2.line(frame, (0, roi_y), (w, roi_y), (80, 180, 255), 1)
    if path is not None:
        cv2.circle(frame, (path["visible_center"], y_look), 6, (255, 255, 255), -1)
    if path_center is not None:
        cv2.circle(frame, (path_center, y_look), 9, (0, 255, 255), -1)
        cv2.line(frame, (f_center, y_look), (path_center, y_look), (0, 255, 255), 3)

    cv2.putText(frame, f"cmd={cmd} err={err} speed={auto_speed_scale:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, lane_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if stop_detected:
        auto_command = "stop"
        auto_speed_scale = 0.0
        if autonomous_enabled:
            autonomous_enabled = False
            lane_status = "Stop Line Reached"
    else:
        auto_command = cmd
    return frame


@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, latest_frame_raw
    data = await request.json()
    frame_b64 = data.get("frame")
    if not frame_b64:
        return {"status": "no frame"}
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "bad frame"}
    loop = asyncio.get_running_loop()
    processed = await loop.run_in_executor(None, process_frame, frame)
    ok, jpeg = cv2.imencode(".jpg", processed)
    if not ok:
        return {"status": "encode failed"}
    # PERF: lock guards globals written here and read by streaming endpoints
    with _lock:
        latest_frame = jpeg.tobytes()
        latest_frame_raw = frame_bytes
    return {"status": "ok", "auto_command": auto_command}


@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            with _lock:
                frame = latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_raw")
async def video_feed_raw():
    async def stream():
        while True:
            with _lock:
                frame = latest_frame_raw
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui.html")
    try:
        with open(html_path) as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>gui.html not found</h1>", status_code=404)


@app.get("/status")
async def status():
    m_cmd = "stop"
    for k in controls:
        if controls[k]:
            m_cmd = k
            break
    return {
        "controls": controls,
        "manual_command": m_cmd,
        "autonomous": autonomous_enabled,
        "auto_command": auto_command,
        "auto_error": auto_error,
        "auto_speed": auto_speed_scale,
        "lane_status": lane_status,
        "martian_detected": martian_detected,
        "martian_confidence": martian_confidence,
        "martian_bbox": martian_bbox,
        "paper_ball_detected": paper_ball_detected,
        "paper_ball_confidence": paper_ball_confidence,
        "paper_ball_bbox": paper_ball_bbox,
        "avoidance_phase": avoidance_phase,
        "paper_rearm_required": paper_rearm_required,
        "robot_state": robot_state,
    }


@app.post("/{direction}")
async def move(direction: str):
    global autonomous_enabled, auto_command, auto_speed_scale, martian_stop_latched, robot_state, avoidance_phase, avoidance_counter, paper_rearm_required, paper_clear_frames
    if direction not in controls:
        return {"error": "invalid direction"}
    autonomous_enabled = False
    auto_command = "stop"
    auto_speed_scale = 0.0
    martian_stop_latched = False
    avoidance_phase = "idle"
    avoidance_counter = 0
    paper_rearm_required = False
    paper_clear_frames = 0
    robot_state = "MANUAL"
    for k in controls:
        controls[k] = False
    controls[direction] = True
    return {direction: True, "autonomous": autonomous_enabled}
