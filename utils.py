"""
utils.py — Core computer-vision utilities for Driver Drowsiness Detection.

Compatible with mediapipe >= 0.10.x (Tasks API — no mp.solutions required).

Handles:
- Eye Aspect Ratio (EAR) calculation
- Landmark index extraction for eyes (MediaPipe FaceLandmarker 478-point mesh)
- Drawing eye contours on frames
"""

import numpy as np
import cv2
import os
import urllib.request

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# EAR below this value = eyes considered closed
EAR_THRESHOLD = 0.22

# MediaPipe FaceLandmarker (478 points) — eye landmark indices
# Same geometry as the old FaceMesh 468-point model
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]

LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33,   7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

# Model file for MediaPipe Tasks FaceLandmarker
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ──────────────────────────────────────────────
# Model auto-download
# ──────────────────────────────────────────────

def ensure_model(dest_dir=None):
    """
    Download face_landmarker.task if not already present.
    Returns the absolute path to the model file.
    """
    if dest_dir is None:
        dest_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dest_dir, MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"[utils] Downloading FaceLandmarker model (~6 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, path, _dl_progress)
            print(f"\n[utils] Model saved to: {path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}\n"
                f"Please download manually from:\n  {MODEL_URL}\n"
                f"and place it next to main.py as '{MODEL_FILENAME}'"
            )
    return path


def _dl_progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\r  Downloading... {pct}%  ({downloaded // 1024} KB)", end="", flush=True)


# ──────────────────────────────────────────────
# EAR Computation
# ──────────────────────────────────────────────

def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_points):
    """
    Compute Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_points: list of 6 (x, y) tuples
    Returns:
        float EAR value
    """
    if len(eye_points) < 6:
        return 0.0
    A = _euclidean(eye_points[1], eye_points[5])
    B = _euclidean(eye_points[2], eye_points[4])
    C = _euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    return round((A + B) / (2.0 * C), 4)


def landmarks_to_pixels(face_landmarks, indices, img_w, img_h):
    """
    Convert normalised MediaPipe NormalizedLandmark objects to pixel coords.

    Works with both old FaceMesh and new FaceLandmarker result formats:
      - Tasks API:  face_landmarks is a list of NormalizedLandmark objects
      - Solutions API (removed): similar layout

    Args:
        face_landmarks: list-like of NormalizedLandmark (x, y, z in [0,1])
        indices: list of int landmark indices
        img_w, img_h: frame width and height in pixels
    Returns:
        list of (px, py) tuples
    """
    pts = []
    for idx in indices:
        lm = face_landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    return pts


def compute_avg_ear(face_landmarks, img_w, img_h):
    """
    Compute average EAR across both eyes.

    Returns:
        (avg_ear, left_pts, right_pts)
    """
    left_pts  = landmarks_to_pixels(face_landmarks, LEFT_EYE_INDICES, img_w, img_h)
    right_pts = landmarks_to_pixels(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)
    avg = (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0
    return round(avg, 4), left_pts, right_pts


# ──────────────────────────────────────────────
# Drawing Utilities
# ──────────────────────────────────────────────

def draw_eye_contours(frame, face_landmarks, img_w, img_h, drowsy=False):
    color     = (0, 0, 255) if drowsy else (0, 255, 120)
    thickness = 2 if drowsy else 1
    for contour_indices in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
        pts = landmarks_to_pixels(face_landmarks, contour_indices, img_w, img_h)
        pts_array = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts_array], isClosed=True,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return frame


def draw_ear_text(frame, ear, drowsy=False, fps=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    ear_color = (0, 255, 120) if not drowsy else (0, 80, 255)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2, cv2.LINE_AA)
    status_text = "DROWSY!" if drowsy else "ALERT"
    status_color = (0, 80, 255) if drowsy else (0, 255, 120)
    cv2.putText(frame, status_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def draw_no_face(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, "No Face Detected", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
    return frame
