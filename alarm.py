"""
alarm.py — Alarm system for Driver Drowsiness Detection.

Generates a WAV alarm file programmatically (no external file needed)
and provides start/stop controls via pygame in a background thread.
"""

import os
import wave
import struct
import math
import threading
import pygame


# ──────────────────────────────────────────────
# WAV Generation
# ──────────────────────────────────────────────

ALARM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm.wav")


def generate_alarm_wav(filepath=ALARM_FILE):
    """
    Programmatically generate an alarm WAV file using pure Python.
    Creates a 3-second urgent multi-tone beep (880 Hz + 1100 Hz alternating).
    """
    if os.path.exists(filepath):
        return  # Already generated

    sample_rate = 44100
    duration = 3.0          # seconds
    num_samples = int(sample_rate * duration)
    num_channels = 1
    sampwidth = 2           # 16-bit
    amplitude = 28000

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Alternate between two frequencies every 0.25s for urgency
        freq = 1100 if int(t / 0.25) % 2 == 0 else 880
        # Slight fade-in for first 0.05s
        fade = min(1.0, t / 0.05)
        value = int(amplitude * fade * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack('<h', value))

    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(samples))


# ──────────────────────────────────────────────
# Alarm Controller
# ──────────────────────────────────────────────

class AlarmSystem:
    """
    Thread-safe alarm controller using pygame mixer.
    Plays alarm.wav in a loop when triggered; stops on demand.
    """

    def __init__(self, alarm_path=ALARM_FILE):
        self.alarm_path = alarm_path
        self._lock = threading.Lock()
        self._playing = False
        self._muted = False
        self._initialized = False
        self._thread = None

        # Generate the WAV file if it doesn't exist
        generate_alarm_wav(self.alarm_path)
        self._init_pygame()

    def _init_pygame(self):
        """Initialize pygame mixer (safe to call multiple times)."""
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._initialized = True
        except Exception as e:
            print(f"[AlarmSystem] pygame init failed: {e}")
            self._initialized = False

    def start(self):
        """Start playing alarm (non-blocking). Has no effect if already playing."""
        with self._lock:
            if self._playing or self._muted:
                return
            self._playing = True

        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self):
        """Internal: load and loop alarm sound."""
        if not self._initialized:
            return
        try:
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.set_volume(0.85)
            pygame.mixer.music.play(loops=-1)  # loop indefinitely

            while True:
                with self._lock:
                    if not self._playing:
                        pygame.mixer.music.stop()
                        break
                pygame.time.wait(100)
        except Exception as e:
            print(f"[AlarmSystem] playback error: {e}")

    def stop(self):
        """Stop the alarm immediately."""
        with self._lock:
            if not self._playing:
                return
            self._playing = False
        # Wait briefly for the thread to stop the mixer
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def is_playing(self):
        with self._lock:
            return self._playing

    def set_muted(self, muted: bool):
        """Mute or unmute. If muted while playing, stops immediately."""
        with self._lock:
            self._muted = muted
            if muted and self._playing:
                self._playing = False
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

    def is_muted(self):
        with self._lock:
            return self._muted

    def cleanup(self):
        """Release pygame resources."""
        self.stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
