"""
main.py — Entry point for Driver Drowsiness Detection System.

Compatible with mediapipe >= 0.10.x using the Tasks API (FaceLandmarker).

Architecture:
  - Main thread:       CustomTkinter UI event loop
  - Detection thread:  OpenCV + MediaPipe FaceLandmarker + EAR computation
"""

import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from ui import DrowsinessApp
from utils import (
    compute_avg_ear,
    draw_eye_contours,
    draw_ear_text,
    draw_no_face,
    EAR_THRESHOLD,
    ensure_model,
)
from alarm import AlarmSystem


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

TARGET_FPS        = 30
EAR_AVG_BUFFER    = 3
EYE_CLOSE_SECONDS = 2.0    # seconds eyes must be closed before alarm
ALARM_COOLDOWN    = 3.0    # seconds between re-triggers


# ──────────────────────────────────────────────
# Detection Engine
# ──────────────────────────────────────────────

class DetectionEngine:
    """
    Runs in a background daemon thread.
    Uses the mediapipe Tasks FaceLandmarker API (works with mediapipe 0.10+).
    """

    def __init__(self, app: DrowsinessApp, alarm: AlarmSystem, model_path: str):
        self.app        = app
        self.alarm      = alarm
        self.model_path = model_path

        self._stop_event    = threading.Event()
        self._thread        = None

        # State
        self._eye_closed_since = None   # timestamp when eyes first closed
        self._ear_buffer       = []
        self._drowsy           = False
        self._last_alarm_ts    = 0.0
        self._fps_counter      = 0
        self._fps_last_time    = time.time()
        self._fps              = 0.0

    # ── Public control ───────────────────────────

    def start(self):
        self._stop_event.clear()
        self._eye_closed_since = None
        self._ear_buffer       = []
        self._drowsy           = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DetectionThread")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ── Detection loop ───────────────────────────

    def _run(self):
        # Build FaceLandmarker with Tasks API
        base_opts = mp_python.BaseOptions(model_asset_path=self.model_path)
        options   = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            # Webcam unavailable — update UI safely via the queue
            self.app._schedule_update(
                self.app._update_status,
                "Camera not found",
                "#DA3633",
                "Check webcam connection and try again"
            )
            detector.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        self.app.set_running(True)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)      # mirror for natural feel
            h, w  = frame.shape[:2]

            # ── FPS counter ───────────────────────
            self._fps_counter += 1
            now     = time.time()
            elapsed = now - self._fps_last_time
            if elapsed >= 1.0:
                self._fps           = self._fps_counter / elapsed
                self._fps_counter   = 0
                self._fps_last_time = now

            # ── MediaPipe inference (Tasks API) ───
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            result = detector.detect(mp_image)

            if result.face_landmarks:
                # result.face_landmarks[0] is a list of NormalizedLandmark
                face_lms = result.face_landmarks[0]

                # Compute EAR
                ear, left_pts, right_pts = compute_avg_ear(face_lms, w, h)

                # Rolling average to reduce jitter
                self._ear_buffer.append(ear)
                if len(self._ear_buffer) > EAR_AVG_BUFFER:
                    self._ear_buffer.pop(0)
                avg_ear  = sum(self._ear_buffer) / len(self._ear_buffer)
                eyes_open = avg_ear > EAR_THRESHOLD

                # Push EAR to UI
                self.app.update_ear(avg_ear)

                # Draw landmarks on frame
                frame = draw_eye_contours(frame, face_lms, w, h,
                                          drowsy=self._drowsy)

                # Time-based eye closure detection
                now = time.time()
                if not eyes_open:
                    if self._eye_closed_since is None:
                        self._eye_closed_since = now        # start timer
                    elif (not self._drowsy
                            and now - self._eye_closed_since >= EYE_CLOSE_SECONDS
                            and now - self._last_alarm_ts > ALARM_COOLDOWN):
                        self._trigger_drowsy()              # 2 sec elapsed
                else:
                    self._eye_closed_since = None           # reset timer
                    if self._drowsy:
                        self._clear_drowsy()

                frame = draw_ear_text(frame, avg_ear,
                                      drowsy=self._drowsy, fps=self._fps)
                self.app.update_monitoring_status(eyes_open)
                self.app.set_no_face(False)

            else:
                # No face detected
                frame = draw_no_face(frame)
                self.app.update_ear(0.0)
                self.app.set_no_face(True)
                if self._drowsy:
                    self._clear_drowsy()

            # ── Push frame to UI ─────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            self.app.update_frame(pil_img)

        # ── Cleanup ──────────────────────────────
        cap.release()
        detector.close()
        self.alarm.stop()
        self.app.set_running(False)
        self.app.set_drowsy(False)

    def _trigger_drowsy(self):
        self._drowsy           = True
        self._last_alarm_ts    = time.time()
        self._eye_closed_since = None
        self.alarm.start()
        self.app.set_drowsy(True)

    def _clear_drowsy(self):
        self._drowsy           = False
        self._eye_closed_since = None
        self.alarm.stop()
        self.app.set_drowsy(False)


# ──────────────────────────────────────────────
# Application Bootstrap
# ──────────────────────────────────────────────

def main():
    # Ensure model is present (downloads ~6 MB on first run)
    print("[main] Checking FaceLandmarker model...")
    model_path = ensure_model()
    print(f"[main] Model ready: {model_path}")

    alarm  = AlarmSystem()
    app    = DrowsinessApp()
    engine = DetectionEngine(app, alarm, model_path)

    # Wire UI callbacks
    def on_start():
        engine.start()

    def on_stop():
        engine.stop()

    def on_mute(muted):
        alarm.set_muted(muted)

    app.on_start       = on_start
    app.on_stop        = on_stop
    app.on_mute_toggle = on_mute

    # Start UI (blocking main thread)
    app.mainloop()

    # Cleanup after window closes
    engine.stop()
    alarm.cleanup()


if __name__ == "__main__":
    main()
