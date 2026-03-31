# 🚗 Driver Drowsiness Detection System

A real-time **driver drowsiness detection application** built using Python, OpenCV, and MediaPipe. The system monitors a driver's face via webcam and detects eye closure using **Eye Aspect Ratio (EAR)**. If the eyes remain closed for more than **2 seconds**, it triggers an alarm and displays a visual warning.

---

## 🎥 Demo

<!-- Add your demo GIF or video here -->
<img width="1920" height="1080" alt="Screenshot 2026-03-31 113912" src="https://github.com/user-attachments/assets/4b5d81f4-d296-4b25-8aa9-444a28da940c" />
<img width="1920" height="1080" alt="Screenshot 2026-03-31 113828" src="https://github.com/user-attachments/assets/114501cf-1107-46b7-8f1d-da8042a5441c" />
<img width="1920" height="1080" alt="Screenshot 2026-03-31 113847" src="https://github.com/user-attachments/assets/45154bc7-8480-4c94-9735-1f6e9b63e716" />



---

## 🚀 Features

* 🎥 Real-time webcam monitoring
* 👁️ Eye Aspect Ratio (EAR) based detection
* 🔔 Audio alarm (auto-generated)
* 🖥️ Modern dark-mode GUI (CustomTkinter)
* 📊 Live EAR meter with color indicator
* 🟢 / 🔴 Status indicator (Eyes Open / Drowsy)
* 📅 Alert history with timestamps
* 🔇 Sound ON/OFF toggle
* ⏱️ Session timer + drowsiness counter
* ✅ No external model files required

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Pygame
* CustomTkinter
* Pillow

---

## 📂 Project Structure

```bash
Driver-Drowsiness-Detection/
├── main.py              # Main logic
├── ui.py                # GUI (CustomTkinter)
├── utils.py             # EAR + helper functions
├── alarm.py             # Alarm controller
├── alarm.wav            # Generated sound
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)

EAR determines whether eyes are open or closed:

```
(||p2 - p6|| + ||p3 - p5||)
EAR = ---------------------
       2 × ||p1 - p4||
```

* **EAR > 0.27** → Eyes Open
* **EAR < 0.22** → Eyes Closed

If EAR stays below threshold for ~2 seconds → 🚨 Alert triggered.

---

## 📋 Requirements

* opencv-python
* mediapipe
* numpy
* pygame
* customtkinter
* Pillow

---

## 🔧 Configuration

| Parameter        | Default    | Description           |
| ---------------- | ---------- | --------------------- |
| EAR_THRESHOLD    | 0.22       | Eye closure threshold |
| CONSEC_THRESHOLD | ~60 frames | Time before alert     |
| TARGET_FPS       | 30         | Webcam FPS            |
| ALARM_COOLDOWN   | 3s         | Delay between alerts  |

---

## ⚡ Challenges

* Handling low-light conditions
* Reducing false positives
* Maintaining real-time performance

---

## 🔮 Future Improvements

* Mobile app integration
* AI-based accuracy improvements
* Custom alert settings
* Cloud logging

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Portfolio project demonstrating real-time computer vision, facial landmark detection, and GUI development.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
