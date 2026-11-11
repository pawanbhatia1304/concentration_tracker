
import cv2
import mediapipe as mp
import numpy as np
import av
import time
import csv
import io
import threading

from datetime import datetime
from collections import deque

from streamlit_webrtc import VideoProcessorBase

from core.feature_extraction import (
    LEFT_EYE,
    RIGHT_EYE,
    calculate_ear,
    calculate_gaze_x,
    estimate_head_pose,
    angle_difference
)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class ConcentrationProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Calibration
        self.calibration_frames = 60
        self.calibration_samples = []
        self.baseline = None

        # Thresholds
        self.blink_threshold = 0.20
        self.gaze_threshold = 0.12
        self.head_threshold = 15.0
        self.prolonged_closure_seconds = 1.5

        # Eye tracking
        self.eye_closed_start = None
        self.last_blink_time = None
        self.blink_count = 0
        self.prolonged_closure = False

        # Score smoothing
        self.smoothed_score = None
        self.smoothing_factor = 0.25

        # FPS
        self.timestamps = deque(maxlen=30)

        # Session data
        self.session_start = time.time()
        self.last_sample_time = 0
        self.sample_interval = 0.2
        self.rows = []

        # Dashboard metrics
        self.metrics = {
            "score": None,
            "fps": 0,
            "status": "WAITING",
            "ear": 0,
            "gaze": 0,
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
            "blinks": 0,
            "calibrated": False,
            "duration": 0
        }

    # ======================================
    # DASHBOARD DATA
    # ======================================

    def get_metrics(self):

        with self.lock:
            return self.metrics.copy()

    def get_csv(self):

        with self.lock:

            if not self.rows:
                return ""

            output = io.StringIO()

            writer = csv.DictWriter(
                output,
                fieldnames=list(self.rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(self.rows)

            return output.getvalue()

    # ======================================
    # CALIBRATION
    # ======================================

    def calibrate(self, ear, gaze, pose):

        if ear <= self.blink_threshold:
            return False

        self.calibration_samples.append({
            "gaze": gaze,
            "yaw": pose["yaw"],
            "pitch": pose["pitch"],
            "roll": pose["roll"]
        })

        if len(self.calibration_samples) >= self.calibration_frames:

            self.baseline = {
                key: float(np.median([
                    sample[key]
                    for sample in self.calibration_samples
                ]))
                for key in ["gaze", "yaw", "pitch", "roll"]
            }

            return True

        return False

    # ======================================
    # BLINK DETECTION
    # ======================================

    def detect_eye_state(self, ear, now):

        if ear < self.blink_threshold:

            if self.eye_closed_start is None:
                self.eye_closed_start = now

            closed_duration = now - self.eye_closed_start

            if closed_duration >= self.prolonged_closure_seconds:
                self.prolonged_closure = True

                return "PROLONGED CLOSURE", closed_duration

            return "BLINKING", closed_duration

        else:

            if self.eye_closed_start is not None:

                closed_duration = now - self.eye_closed_start

                if 0.05 <= closed_duration < 0.5:
                    self.blink_count += 1
                    self.last_blink_time = now

            self.eye_closed_start = None
            self.prolonged_closure = False

            return "OPEN", 0.0

    # ======================================
    # CONCENTRATION SCORING
    # ======================================

    def calculate_score(
        self,
        ear,
        gaze_deviation,
        yaw_deviation,
        pitch_deviation,
        roll_deviation,
        eye_state
    ):

        # Eye component: 40%
        eye_score = (
            0 if eye_state == "PROLONGED CLOSURE"
            else 100
        )

        # Gaze component: 30%
        gaze_score = (
            100 if gaze_deviation <= self.gaze_threshold
            else 0
        )

        # Head pose component: 30%
        head_deviation = max(
            yaw_deviation,
            pitch_deviation,
            roll_deviation
        )

        head_score = (
            100 if head_deviation <= self.head_threshold
            else 0
        )

        raw_score = (
            eye_score * 0.40
            + gaze_score * 0.30
            + head_score * 0.30
        )

        # Exponential moving average
        if self.smoothed_score is None:

            self.smoothed_score = raw_score

        else:

            self.smoothed_score = (
                self.smoothing_factor * raw_score
                + (1 - self.smoothing_factor)
                * self.smoothed_score
            )

        return round(self.smoothed_score, 1)

    # ======================================
    # CSV RECORDING
    # ======================================

    def record_sample(
        self,
        now,
        ear,
        gaze,
        gaze_deviation,
        yaw,
        pitch,
        roll,
        eye_state,
        score,
        status
    ):

        if now - self.last_sample_time < self.sample_interval:
            return

        self.last_sample_time = now

        row = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(
                now - self.session_start, 2
            ),
            "ear": round(ear, 4),
            "gaze_x": round(gaze, 4),
            "gaze_deviation": round(gaze_deviation, 4),
            "yaw_relative": round(yaw, 3),
            "pitch_relative": round(pitch, 3),
            "roll_relative": round(roll, 3),
            "eye_state": eye_state,
            "blink_count": self.blink_count,
            "visual_attention_score": score,
            "visual_status": status,
            "visual_attention_label": int(status == "ATTENTIVE")
        }

        self.rows.append(row)

    # ======================================
    # MAIN VIDEO PROCESSING
    # ======================================

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        height, width, _ = image.shape

        now = time.perf_counter()

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        status = "NO FACE DETECTED"

        score = None
        ear = 0.0
        gaze = None

        yaw_relative = 0.0
        pitch_relative = 0.0
        roll_relative = 0.0

        eye_state = "UNKNOWN"

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image,
                face,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1
                )
            )

            # EAR
            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            ear = (left_ear + right_ear) / 2

            # Gaze
            gaze = calculate_gaze_x(
                landmarks, width, height
            )

            # Head pose
            pose = estimate_head_pose(
                landmarks, width, height
            )

            if gaze is not None and pose is not None:

                # Calibration
                if self.baseline is None:

                    self.calibrate(ear, gaze, pose)

                    if self.baseline is None:

                        status = (
                            f"CALIBRATING "
                            f"{len(self.calibration_samples)}/"
                            f"{self.calibration_frames}"
                        )

                # Normal monitoring
                if self.baseline is not None:

                    gaze_deviation = abs(
                        gaze - self.baseline["gaze"]
                    )

                    yaw_relative = angle_difference(
                        pose["yaw"], self.baseline["yaw"]
                    )

                    pitch_relative = angle_difference(
                        pose["pitch"], self.baseline["pitch"]
                    )

                    roll_relative = angle_difference(
                        pose["roll"], self.baseline["roll"]
                    )

                    # Eye state
                    eye_state, closure_duration = (
                        self.detect_eye_state(ear, now)
                    )

                    # Score
                    score = self.calculate_score(
                        ear,
                        gaze_deviation,
                        abs(yaw_relative),
                        abs(pitch_relative),
                        abs(roll_relative),
                        eye_state
                    )

                    # Visual status
                    looking_away = (
                        gaze_deviation > self.gaze_threshold
                        or abs(yaw_relative) > self.head_threshold
                        or abs(pitch_relative) > self.head_threshold
                        or abs(roll_relative) > self.head_threshold
                    )

                    if eye_state == "PROLONGED CLOSURE":

                        status = "EYES CLOSED"

                    elif looking_away:

                        status = "LOOKING AWAY"

                    elif eye_state == "BLINKING":

                        status = "BLINKING"

                    else:

                        status = "ATTENTIVE"

                    # Record features
                    self.record_sample(
                        now,
                        ear,
                        gaze,
                        gaze_deviation,
                        yaw_relative,
                        pitch_relative,
                        roll_relative,
                        eye_state,
                        score,
                        status
                    )

                    # Display measurements
                    display_items = [
                        (f"EAR: {ear:.3f}", (0, 255, 255)),
                        (f"Gaze: {gaze:.2f}", (255, 255, 0)),
                        (f"Yaw: {yaw_relative:+.1f}", (255, 255, 0)),
                        (f"Pitch: {pitch_relative:+.1f}", (255, 255, 0)),
                        (f"Roll: {roll_relative:+.1f}", (255, 255, 0)),
                        (f"Score: {score:.0f}%", (0, 255, 0))
                    ]

                    for i, (text, color) in enumerate(display_items):

                        cv2.putText(
                            image,
                            text,
                            (20, 35 + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2
                        )

                    status_color = (
                        (0, 255, 0)
                        if status == "ATTENTIVE"
                        else (0, 165, 255)
                    )

                    cv2.putText(
                        image,
                        status,
                        (20, 245),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2
                    )

            else:

                status = "FEATURE EXTRACTION FAILED"

        # FPS
        self.timestamps.append(now)

        fps = 0.0

        if len(self.timestamps) > 1:

            elapsed = (
                self.timestamps[-1] - self.timestamps[0]
            )

            if elapsed > 0:
                fps = (len(self.timestamps) - 1) / elapsed

        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (width - 160, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # Update dashboard safely
        with self.lock:

            self.metrics = {
                "score": score,
                "fps": round(fps, 1),
                "status": status,
                "ear": round(ear, 3),
                "gaze": round(gaze, 3) if gaze is not None else 0,
                "yaw": round(yaw_relative, 2),
                "pitch": round(pitch_relative, 2),
                "roll": round(roll_relative, 2),
                "blinks": self.blink_count,
                "calibrated": self.baseline is not None,
                "duration": round(time.time() - self.session_start, 1)
            }

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def __del__(self):

        try:
            self.face_mesh.close()
        except Exception:
            pass
