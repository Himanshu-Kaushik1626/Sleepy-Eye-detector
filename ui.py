"""
ui.py — Modern CustomTkinter Dashboard for Driver Drowsiness Detection.

Dashboard Layout:
  ┌────────────────────────────────────────────────────────┐
  │  🚗  Driver Drowsiness Detection System          [─][□][✕]│
  ├─────────────────────────────┬──────────────────────────┤
  │                             │  Status Indicator         │
  │       LIVE VIDEO FEED       │  EAR Meter                │
  │       (webcam / OpenCV)     │  Stats Panel              │
  │                             │  Alert History            │
  │                             │  Controls                 │
  ├─────────────────────────────┴──────────────────────────┤
  │  Status Bar                                             │
  └────────────────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import font as tkfont
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import threading
import datetime
import math
import time
import queue


# ──────────────────────────────────────────────
# Theme / Color Palette
# ──────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_dark":       "#0D1117",
    "bg_panel":      "#161B22",
    "bg_card":       "#1C2128",
    "bg_card2":      "#21262D",
    "accent_blue":   "#1F6FEB",
    "accent_green":  "#2EA043",
    "accent_red":    "#DA3633",
    "accent_orange": "#D29922",
    "text_primary":  "#F0F6FC",
    "text_secondary":"#8B949E",
    "border":        "#30363D",
    "glow_green":    "#39D353",
    "glow_red":      "#FF4444",
}


# ──────────────────────────────────────────────
# Helper: Placeholder frame (no feed)
# ──────────────────────────────────────────────

def _make_placeholder_frame(w=640, h=480, text="Camera Off"):
    """Create a dark placeholder image with centered text."""
    img = Image.new("RGB", (w, h), color=(13, 17, 23))
    draw = ImageDraw.Draw(img)
    # grid lines feel
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill=(25, 30, 38), width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill=(25, 30, 38), width=1)
    # Icon circle
    cx, cy, r = w // 2, h // 2 - 30, 50
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(40, 80, 140), width=2)
    draw.line([(cx - 20, cy - 20), (cx + 20, cy + 20)],
              fill=(100, 40, 40), width=3)
    draw.line([(cx + 20, cy - 20), (cx - 20, cy + 20)],
              fill=(100, 40, 40), width=3)
    # Text
    draw.text((w // 2, h // 2 + 40), text,
              fill=(100, 110, 120), anchor="mm")
    return img


# ──────────────────────────────────────────────
# Main App Window
# ──────────────────────────────────────────────

class DrowsinessApp(ctk.CTk):

    # Pulse animation steps (red glow)
    PULSE_STEPS = 20

    def __init__(self):
        super().__init__()

        self.title("Driver Drowsiness Detection System")
        self.geometry("1180x720")
        self.minsize(960, 620)
        self.configure(fg_color=COLORS["bg_dark"])

        # ── State Variables ──
        self._running = False
        self._muted = False
        self._drowsy = False
        self._ear_value = 0.0
        self._drowsy_count = 0
        self._alert_history = []      # list of (time_str, message)
        self._pulse_step = 0
        self._pulse_dir = 1
        self._pulse_active = False
        self._start_time = None
        self._frame_count = 0
        self._no_face = False

        # Thread-safe UI update queue
        self._ui_queue = queue.Queue()

        # Callbacks (set by main.py)
        self.on_start = None
        self.on_stop = None

        # ── Build UI ──
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

        # Kick off animation + queue-processor loops
        self._animate()
        self._process_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────
    # Layout: Header
    # ──────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                               corner_radius=0, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Left: logo + title
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=20, pady=8, fill="y")

        self._title_canvas = tk.Canvas(left, width=36, height=36,
                                       bg=COLORS["bg_panel"], highlightthickness=0)
        self._title_canvas.pack(side="left", padx=(0, 12))
        self._draw_logo(self._title_canvas)

        title_frame = ctk.CTkFrame(left, fg_color="transparent")
        title_frame.pack(side="left")
        ctk.CTkLabel(title_frame, text="Driver Drowsiness Detection",
                     font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="Real-time eye monitoring system",
                     font=ctk.CTkFont(family="Segoe UI", size=11),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        # Right: time + sound toggle
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=20, pady=8, fill="y")

        self._clock_label = ctk.CTkLabel(right, text="",
                                          font=ctk.CTkFont(family="Segoe UI Mono", size=13),
                                          text_color=COLORS["text_secondary"])
        self._clock_label.pack(side="right", padx=(16, 0))

        self._sound_btn = ctk.CTkButton(
            right, text="🔊 Sound ON", width=110, height=32,
            fg_color=COLORS["bg_card2"], hover_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            font=ctk.CTkFont(size=12), corner_radius=8,
            command=self._toggle_sound
        )
        self._sound_btn.pack(side="right")

    def _draw_logo(self, canvas):
        """Draw a simple eye / car logo on canvas."""
        canvas.create_oval(4, 4, 32, 32, outline=COLORS["accent_blue"], width=2)
        canvas.create_oval(12, 12, 24, 24, fill=COLORS["accent_blue"], outline="")
        canvas.create_oval(15, 15, 21, 21, fill=COLORS["bg_panel"], outline="")

    # ──────────────────────────────────────────
    # Layout: Main Area (video | right panel)
    # ──────────────────────────────────────────

    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=(12, 8))
        # minsize prevents either column ever collapsing to zero
        main.columnconfigure(0, weight=3, minsize=480)
        main.columnconfigure(1, weight=2, minsize=340)
        main.rowconfigure(0, weight=1)

        self._build_video_panel(main)
        self._build_right_panel(main)

    # ── Video Panel ──

    def _build_video_panel(self, parent):
        wrapper = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                                corner_radius=12, border_width=1,
                                border_color=COLORS["border"])
        wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        # Prevent wrapper from resizing to match its children (image label)
        wrapper.grid_propagate(False)
        wrapper.rowconfigure(1, weight=1)
        wrapper.columnconfigure(0, weight=1)

        # Top bar
        top_bar = ctk.CTkFrame(wrapper, fg_color=COLORS["bg_card2"],
                                corner_radius=0, height=36)
        top_bar.grid(row=0, column=0, sticky="ew", padx=1)
        top_bar.pack_propagate(False)

        ctk.CTkLabel(top_bar, text=" 📹  Live Camera Feed",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=12)

        self._rec_dot = ctk.CTkLabel(top_bar, text="● REC",
                                      font=ctk.CTkFont(size=11, weight="bold"),
                                      text_color="#555555")
        self._rec_dot.pack(side="right", padx=12)

        # Video label — fixed display size, never drives layout
        self._VIDEO_W = 640
        self._VIDEO_H = 440
        placeholder = _make_placeholder_frame(
            self._VIDEO_W, self._VIDEO_H, "Click START to begin monitoring")
        self._ph_tk = ctk.CTkImage(
            light_image=placeholder, dark_image=placeholder,
            size=(self._VIDEO_W, self._VIDEO_H))
        self._video_label = ctk.CTkLabel(
            wrapper, image=self._ph_tk, text="",
            fg_color=COLORS["bg_dark"])
        self._video_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Drowsy overlay label (hidden by default)
        self._drowsy_overlay = ctk.CTkLabel(
            wrapper, text="⚠  DROWSINESS DETECTED!",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#FF4040",
            fg_color="transparent"
        )
        # Store wrapper ref for overlay placement
        self._video_wrapper = wrapper

    # ── Right Panel ──

    def _build_right_panel(self, parent):
        # Outer frame anchors the column — does NOT propagate child sizes
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        # Inner scrollable frame sits inside the outer anchor
        right = ctk.CTkScrollableFrame(
            outer,
            fg_color="transparent",
            scrollbar_fg_color=COLORS["border"],
            scrollbar_button_color=COLORS["bg_card2"],
        )
        right.grid(row=0, column=0, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self._build_status_card(right)
        self._build_ear_card(right)
        self._build_stats_card(right)
        self._build_controls_card(right)
        self._build_history_card(right)

    def _card(self, parent, title, row):
        """Convenience: create a labelled card frame."""
        card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"],
                             corner_radius=10, border_width=1,
                             border_color=COLORS["border"])
        card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        card.columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=title,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=COLORS["text_secondary"]).pack(
                         anchor="w", padx=14, pady=(10, 4))
        return card

    # ── Status Card ──

    def _build_status_card(self, parent):
        card = self._card(parent, "SYSTEM STATUS", row=0)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=(0, 12))

        self._status_dot_canvas = tk.Canvas(inner, width=18, height=18,
                                             bg=COLORS["bg_card"],
                                             highlightthickness=0)
        self._status_dot_canvas.pack(side="left")
        self._status_dot = self._status_dot_canvas.create_oval(
            2, 2, 16, 16, fill=COLORS["text_secondary"], outline="")

        self._status_label = ctk.CTkLabel(inner, text="Idle — Not monitoring",
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           text_color=COLORS["text_secondary"])
        self._status_label.pack(side="left", padx=(10, 0))

        # Sub-info
        sub = ctk.CTkFrame(card, fg_color="transparent")
        sub.pack(fill="x", padx=14, pady=(0, 10))

        self._sub_status = ctk.CTkLabel(sub, text="Start the system to begin monitoring",
                                         font=ctk.CTkFont(size=11),
                                         text_color=COLORS["text_secondary"])
        self._sub_status.pack(anchor="w")

    # ── EAR Card ──

    def _build_ear_card(self, parent):
        card = self._card(parent, "EYE ASPECT RATIO (EAR)", row=1)

        ear_row = ctk.CTkFrame(card, fg_color="transparent")
        ear_row.pack(fill="x", padx=14, pady=(0, 6))

        self._ear_value_label = ctk.CTkLabel(ear_row, text="0.000",
                                              font=ctk.CTkFont(family="Segoe UI Mono",
                                                               size=28, weight="bold"),
                                              text_color=COLORS["text_primary"])
        self._ear_value_label.pack(side="left")

        ear_info = ctk.CTkFrame(ear_row, fg_color="transparent")
        ear_info.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(ear_info, text="Threshold: 0.22",
                     font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")
        self._ear_state_label = ctk.CTkLabel(ear_info, text="—",
                                              font=ctk.CTkFont(size=11, weight="bold"),
                                              text_color=COLORS["text_secondary"])
        self._ear_state_label.pack(anchor="w")

        self._ear_bar = ctk.CTkProgressBar(card, width=220, height=12,
                                            progress_color=COLORS["accent_green"],
                                            fg_color=COLORS["bg_card2"],
                                            corner_radius=6)
        self._ear_bar.pack(fill="x", padx=14, pady=(0, 10))
        self._ear_bar.set(0)

    # ── Stats Card ──

    def _build_stats_card(self, parent):
        card = self._card(parent, "SESSION STATISTICS", row=2)

        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure((0, 1), weight=1)

        def _stat(parent, label, col):
            f = ctk.CTkFrame(parent, fg_color=COLORS["bg_card2"],
                             corner_radius=8)
            f.grid(row=0, column=col, sticky="ew", padx=(0, 6) if col == 0 else (6, 0))
            lbl = ctk.CTkLabel(f, text="0",
                               font=ctk.CTkFont(size=24, weight="bold"),
                               text_color=COLORS["text_primary"])
            lbl.pack(pady=(10, 2))
            ctk.CTkLabel(f, text=label,
                         font=ctk.CTkFont(size=10),
                         text_color=COLORS["text_secondary"]).pack(pady=(0, 10))
            return lbl

        self._drowsy_count_label = _stat(grid, "Drowsy Events", 0)
        self._session_time_label = _stat(grid, "Session (s)", 1)

    # ── Controls Card ──

    def _build_controls_card(self, parent):
        card = self._card(parent, "CONTROLS", row=3)

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 12))
        btn_row.columnconfigure((0, 1), weight=1)

        self._start_btn = ctk.CTkButton(
            btn_row, text="▶  Start Monitoring",
            fg_color=COLORS["accent_green"], hover_color="#3FB350",
            text_color="white", font=ctk.CTkFont(size=13, weight="bold"),
            height=42, corner_radius=8,
            command=self._handle_start
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            btn_row, text="■  Stop",
            fg_color=COLORS["bg_card2"], hover_color=COLORS["border"],
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42, corner_radius=8,
            state="disabled",
            command=self._handle_stop
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    # ── Alert History Card ──

    def _build_history_card(self, parent):
        card = self._card(parent, "ALERT HISTORY", row=4)

        self._history_frame = ctk.CTkFrame(card, fg_color="transparent")
        self._history_frame.pack(fill="x", padx=14, pady=(0, 10))

        self._empty_history_label = ctk.CTkLabel(
            self._history_frame,
            text="No alerts yet",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self._empty_history_label.pack(anchor="w")

    # ──────────────────────────────────────────
    # Layout: Status Bar
    # ──────────────────────────────────────────

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                            corner_radius=0, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._statusbar_label = ctk.CTkLabel(
            bar, text="Ready — Waiting to start",
            font=ctk.CTkFont(family="Segoe UI Mono", size=10),
            text_color=COLORS["text_secondary"]
        )
        self._statusbar_label.pack(side="left", padx=14)

        ctk.CTkLabel(bar,
                     text="MediaPipe FaceMesh · EAR Detection · v1.0.0",
                     font=ctk.CTkFont(family="Segoe UI Mono", size=10),
                     text_color=COLORS["border"]).pack(side="right", padx=14)

    # ──────────────────────────────────────────
    # Public API (called from main.py)
    # ──────────────────────────────────────────

    # ──────────────────────────────────────────
    # Thread-safe queue bridge
    # ──────────────────────────────────────────

    def _schedule_update(self, fn, *args):
        """Queue a UI update to be executed on the main thread."""
        self._ui_queue.put((fn, args))

    def _process_queue(self):
        """Drain the UI update queue — always runs on the main thread."""
        try:
            while True:
                fn, args = self._ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception as e:
                    print(f"[UI queue] error: {e}")
        except queue.Empty:
            pass
        self.after(10, self._process_queue)

    # ──────────────────────────────────────────
    # Public API (called from any thread)
    # ──────────────────────────────────────────

    def update_frame(self, pil_image):
        """Push a new PIL image to the video label (call from any thread)."""
        self._frame_count += 1
        self._schedule_update(self._set_frame, pil_image)

    def _set_frame(self, pil_image):
        """Resize incoming frame to fixed video dimensions and update label."""
        img = pil_image.resize((self._VIDEO_W, self._VIDEO_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=img, dark_image=img,
            size=(self._VIDEO_W, self._VIDEO_H))
        self._video_label.configure(image=ctk_img)
        self._video_label.image = ctk_img   # prevent GC

    def update_ear(self, ear_value: float):
        """Update EAR display (thread-safe)."""
        self._ear_value = ear_value
        self._schedule_update(self._refresh_ear)

    def _refresh_ear(self):
        ear = self._ear_value
        self._ear_value_label.configure(text=f"{ear:.3f}")

        # Normalise bar: EAR range approx 0.0 → 0.40
        progress = min(1.0, ear / 0.40)
        self._ear_bar.set(progress)

        from utils import EAR_THRESHOLD
        if ear > EAR_THRESHOLD + 0.05:
            color = COLORS["accent_green"]
            state = "Eyes Open"
        elif ear > EAR_THRESHOLD:
            color = COLORS["accent_orange"]
            state = "Closing..."
        else:
            color = COLORS["accent_red"]
            state = "Eyes Closed"

        self._ear_bar.configure(progress_color=color)
        self._ear_state_label.configure(text=state,
                                         text_color=color if ear > 0.01 else COLORS["text_secondary"])

    def set_drowsy(self, is_drowsy: bool):
        """Show / hide drowsiness alert (thread-safe)."""
        if self._drowsy == is_drowsy:
            return
        self._drowsy = is_drowsy
        self._schedule_update(self._refresh_drowsy_state, is_drowsy)

    def _refresh_drowsy_state(self, is_drowsy):
        if is_drowsy:
            self._drowsy_count += 1
            self._drowsy_count_label.configure(text=str(self._drowsy_count))
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._add_alert(ts, "Drowsiness detected!")
            self._pulse_active = True
        else:
            self._pulse_active = False
            self._drowsy_overlay.place_forget()

    def set_no_face(self, no_face: bool):
        """Update status when no face is detected."""
        self._no_face = no_face
        self._schedule_update(self._refresh_no_face, no_face)

    def _refresh_no_face(self, no_face):
        if no_face:
            self._update_status("No face detected", COLORS["accent_orange"],
                                 "Move into camera frame")

    def set_running(self, running: bool):
        """Sync running state UI (thread-safe)."""
        self._schedule_update(self._refresh_running, running)

    def _refresh_running(self, running):
        self._running = running
        if running:
            self._start_btn.configure(state="disabled",
                                       fg_color=COLORS["border"])
            self._stop_btn.configure(state="normal",
                                      fg_color=COLORS["accent_red"],
                                      text_color="white", text="■  Stop")
            self._rec_dot.configure(text_color="#DA3633")
            self._start_time = time.time()
            self._update_status("Monitoring Driver", COLORS["accent_green"],
                                 "System active · watching for eye closure")
            self._statusbar_label.configure(
                text="● Monitoring active — detecting drowsiness")
        else:
            self._start_btn.configure(state="normal",
                                       fg_color=COLORS["accent_green"])
            self._stop_btn.configure(state="disabled",
                                      fg_color=COLORS["bg_card2"],
                                      text_color=COLORS["text_secondary"])
            self._rec_dot.configure(text="● REC", text_color="#555555")
            self._pulse_active = False
            self._drowsy_overlay.place_forget()
            self._update_status("Idle — Not monitoring", COLORS["text_secondary"],
                                 "Start the system to begin monitoring")
            self._statusbar_label.configure(text="Stopped")
            self._start_time = None

    def _update_status(self, text, color, sub=""):
        self._status_dot_canvas.itemconfig(self._status_dot, fill=color)
        self._status_label.configure(text=text, text_color=color)
        if sub:
            self._sub_status.configure(text=sub)

    def update_monitoring_status(self, eyes_open: bool):
        """Called each frame when face is detected."""
        self._schedule_update(self._refresh_monitoring_status, eyes_open)

    def _refresh_monitoring_status(self, eyes_open):
        if not self._running:
            return
        if self._drowsy:
            self._update_status("⚠  Drowsiness Detected!", COLORS["accent_red"],
                                 "Please wake up! Alarm active.")
        elif eyes_open:
            self._update_status("Eyes Open  ✓", COLORS["accent_green"],
                                 "Driver is alert")
        else:
            self._update_status("Eyes Closing…", COLORS["accent_orange"],
                                 "Monitoring eye closure duration")

    # ──────────────────────────────────────────
    # Alert History
    # ──────────────────────────────────────────

    def _add_alert(self, ts, message):
        self._alert_history.insert(0, (ts, message))
        if len(self._alert_history) > 8:
            self._alert_history.pop()
        self._rebuild_history()

    def _rebuild_history(self):
        for w in self._history_frame.winfo_children():
            w.destroy()
        if not self._alert_history:
            ctk.CTkLabel(self._history_frame, text="No alerts yet",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_secondary"]).pack(anchor="w")
            return
        for ts, msg in self._alert_history:
            row = ctk.CTkFrame(self._history_frame,
                               fg_color=COLORS["bg_card2"], corner_radius=6)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"⚠ {msg}",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=COLORS["glow_red"]).pack(side="left", padx=8, pady=4)
            ctk.CTkLabel(row, text=ts,
                         font=ctk.CTkFont(family="Segoe UI Mono", size=10),
                         text_color=COLORS["text_secondary"]).pack(side="right", padx=8)

    # ──────────────────────────────────────────
    # Button Handlers
    # ──────────────────────────────────────────

    def _handle_start(self):
        if self.on_start:
            self.on_start()

    def _handle_stop(self):
        if self.on_stop:
            self.on_stop()

    def _toggle_sound(self):
        self._muted = not self._muted
        if self._muted:
            self._sound_btn.configure(text="🔇 Sound OFF",
                                       fg_color=COLORS["accent_red"])
        else:
            self._sound_btn.configure(text="🔊 Sound ON",
                                       fg_color=COLORS["bg_card2"])
        if self.on_mute_toggle:
            self.on_mute_toggle(self._muted)

    def on_mute_toggle(self, muted): pass  # override in main.py

    # ──────────────────────────────────────────
    # Animation Loop
    # ──────────────────────────────────────────

    def _animate(self):
        """60-fps UI animation tick."""
        self._tick_clock()
        self._tick_session_timer()
        self._tick_pulse()
        self.after(16, self._animate)

    def _tick_clock(self):
        self._clock_label.configure(
            text=datetime.datetime.now().strftime("%H:%M:%S"))

    def _tick_session_timer(self):
        if self._start_time and self._running:
            elapsed = int(time.time() - self._start_time)
            self._session_time_label.configure(text=str(elapsed))

    def _tick_pulse(self):
        if not self._pulse_active:
            return

        self._pulse_step += self._pulse_dir
        if self._pulse_step >= self.PULSE_STEPS:
            self._pulse_dir = -1
        elif self._pulse_step <= 0:
            self._pulse_dir = 1

        alpha = self._pulse_step / self.PULSE_STEPS
        # Update drowsy overlay visibility using alpha
        vl = self._video_label
        x = vl.winfo_x() + vl.winfo_width() // 2
        y = vl.winfo_y() + 30

        if alpha > 0.5:
            self._drowsy_overlay.place(
                in_=self._video_wrapper,
                rely=0.04, relx=0.5, anchor="n"
            )
        else:
            self._drowsy_overlay.place_forget()

        # Change status dot color pulsing
        if alpha > 0.5:
            self._status_dot_canvas.itemconfig(self._status_dot,
                                                fill=COLORS["glow_red"])
        else:
            self._status_dot_canvas.itemconfig(self._status_dot,
                                                fill=COLORS["accent_red"])

    # ──────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────

    def _on_close(self):
        if self.on_stop:
            self.on_stop()
        self.after(200, self.destroy)
