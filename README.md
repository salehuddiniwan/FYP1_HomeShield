# 🛡️ HomeShield

**Intelligent ML-powered CCTV surveillance for elderly & child safety.**

HomeShield is a real-time computer vision system that autonomously watches over vulnerable people at home. It detects falls, prolonged inactivity, children entering dangerous areas, and unrecognised people on the premises — and sends instant WhatsApp alerts with annotated snapshots. All inference runs locally; video never leaves the machine.

Built with **YOLOv8-pose** (detection + pose in a single GPU pass), **InsightFace** (face recognition + age), **Flask**, and **OpenCV**.

---

## ✨ Features

### Detection
- **Real-time person detection** — YOLOv8-pose on GPU (~8–15 ms per frame on RTX 4070)
- **17-keypoint pose estimation** — one model does detection AND pose in a single forward pass
- **Rule-based fall detector** — body angle, vertical/horizontal ratio, and hip velocity over a sliding window
- **Inactivity monitoring** — configurable timeout for "person hasn't moved in N seconds"
- **Multi-camera batching** — all active cameras go through the GPU as one batch per tick

### Identity & alerts
- **Face recognition with registered-persons list** — identifies family members by name on the live feed (powered by InsightFace's ArcFace embeddings)
- **Intruder detection** — anyone whose face doesn't match anyone on the list is flagged in red, logged with a face crop, and notified via WhatsApp
- **Live face-detect preview** on the register panel — see a green guide box around the detected face before you click capture
- **Intruder gallery** — see every stranger the system has spotted, with one-click promotion to the registered list
- **WhatsApp alerts** via Twilio — event type, camera name, time, snapshot

### Zones
- **Danger zones** — draw polygons on the camera feed; alerts fire when a child enters (e.g. kitchen stove, staircase)
- **Safe zones** — suppress lying/inactivity alerts inside these areas (e.g. the bed, so sleeping doesn't trigger alarms)
- **Visual zone editor** — click-to-place polygon vertices directly on the camera snapshot

### Platform
- **Web dashboard** — live feeds, event history, zone editor, persons & intruders page, settings
- **Local-only processing** — no cloud dependency, no video upload
- **Graceful GPU fallback** — automatically uses CUDA, Apple MPS, or CPU depending on availability
- **Model hot-swap** — change YOLO size / image size / FP16 toggle in Settings, Stop → Start and the new config is picked up without restarting the process

---

## 📋 Requirements

### Recommended setup (what this is tuned for)
- **GPU**: NVIDIA RTX 30-series or newer, 8 GB+ VRAM
- **CUDA**: 12.1 (matches the pinned PyTorch wheel)
- **Python**: 3.11 (required — see *Python version note* below)
- **RAM**: 16 GB
- **OS**: Windows 10/11, Ubuntu 22.04+, or macOS

### Minimum (CPU-only, no intruder detection)
- Python 3.11
- 8 GB RAM, any Intel i5 / Ryzen 5 or better
- Webcam or any RTSP-capable IP camera

### Python version note
**You must use Python 3.11.** InsightFace does not publish prebuilt wheels for Python 3.12+ on Windows, and compiling from source requires the Microsoft C++ Build Tools. Python 3.11 has a community-maintained prebuilt wheel (see installation section).

---

## ⚡ Installation

### 1. Get Python 3.11

Download from [python.org](https://www.python.org/downloads/windows/) — pick the latest 3.11.x installer. During install:
- ✅ Add python.exe to PATH
- ✅ Install for all users (makes the `py -3.11` launcher available)

### 2. Clone and create a venv

```bash
git clone <your-repo-url> HomeShield
cd HomeShield
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # macOS / Linux
python --version              # should print 3.11.x
python -m pip install --upgrade pip
```

### 3. Install PyTorch (do this BEFORE requirements.txt)

Pick the line that matches your hardware:

```bash
# NVIDIA CUDA 12.1 (RTX 30/40 series — recommended)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# NVIDIA CUDA 11.8 (older cards, GTX 10/16/20 series)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# CPU-only fallback (works on any machine, much slower)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2/M3)
pip install torch==2.5.1 torchvision==0.20.1
```

### 4. Install InsightFace (Windows-specific)

On **Linux/macOS**, `pip install` works directly — skip this step, the next one handles it.

On **Windows**, PyPI doesn't ship a prebuilt wheel, so download the community-maintained one:

1. Go to https://github.com/Gourieff/Assets/tree/main/Insightface
2. Download `insightface-0.7.3-cp311-cp311-win_amd64.whl`
3. Install it:

```bash
pip install path\to\insightface-0.7.3-cp311-cp311-win_amd64.whl
```

### 5. Install the rest

```bash
pip install -r requirements.txt
```

### 6. Configure

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Open `.env` and fill in the values you care about. For a first-run test, the defaults work — Twilio is only needed for real WhatsApp alerts (the system logs alerts to the console otherwise).

### 7. Run

```bash
python app.py
```

You should see:

```
[GPU] CUDA: NVIDIA GeForce RTX 4070  VRAM: 12.0 GB
[INFO] InsightFace ready (detection + ArcFace + age + gender)
[INFO] YOLOv8-pose on cuda (fp16=True): yolov8m-pose.pt
[INFO] Camera started: Camera 1 (0)
 * Running on http://0.0.0.0:5000
```

Open http://localhost:5000 in a browser.

On first run, Ultralytics downloads the YOLOv8 pose weights (~7 MB for nano, up to ~136 MB for XLarge) and InsightFace downloads the `buffalo_l` model pack (~300 MB). These are cached and only download once.

---

## 📸 Camera setup

### Built-in webcam
Use URL `0` (or `1` if you have multiple cameras).

### TP-Link Tapo C200 / C210
In the Tapo app, enable the camera account. Then:
```
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
```

### Android phone as camera (DroidCam, IP Webcam)
```
http://PHONE_IP:4747/video
```

### Video file for testing
Use the absolute path: `C:/videos/fall_test.mp4` (forward slashes on Windows too).

Add cameras in the dashboard under **Settings → Cameras**.

---

## 🎯 Using the system

### Register your family members (before you turn on intruder detection)

Go to the **Persons** page. On the right, the register panel shows a live preview from the selected camera with face detection running in real time. Ask the person to stand in front of the camera and face it directly. When the guide box turns **green** and the label reads "FACE OK", click **Capture from camera**.

**Important**: until you register at least one person, intruder detection is silent by design — it would flag everyone otherwise.

### Set up zones

Go to the **Zones** page:
- Select the camera
- Choose **Danger** (alerts when a child enters — kitchens, staircases) or **Safe** (suppresses lying/inactivity alerts — beds, sofas)
- Click points on the image to define the polygon
- Name it and save

### Watch the live feed

The **Live** tab shows annotated camera feeds. Registered people appear with their name ("Grandma Siti (elderly)"); unrecognized people are labeled "INTRUDER" in red after a few seconds of clear face visibility.

### Handle intruders

When the system flags someone, they appear in the **Intruders seen** section on the Persons page with a cropped face photo. You can:
- **Register** — promote them to a known person (prompts for name + category)
- **Dismiss** — mark as handled (doesn't re-alert unless they appear again later)

---

## ⚙️ Tuning for your hardware

Open **Settings → Model & GPU** and pick based on your GPU:

| GPU                        | YOLO model | Image size | FPS | FP16 |
|----------------------------|-----------|-----------|-----|------|
| RTX 40-series (12 GB+)     | `yolov8l` | 640       | 20  | On   |
| RTX 30-series (8–12 GB)    | `yolov8m` | 640       | 15  | On   |
| GTX 16xx / older GPU       | `yolov8s` | 512       | 10  | Off  |
| CPU only                   | `yolov8n` | 480       | 8   | —    |

**Fall confidence threshold** — 80% default. Lower to catch more falls at the cost of false positives; raise to reduce false alerts.

**Detection confidence** — 45% default. Lower catches partially occluded people (crouched, lying down); raise if you're getting too many cardboard-box-is-a-person alerts.

**Alert cooldown** — 60 s default. Same event won't re-fire for this many seconds on the same camera.

---

## 🏗️ Architecture

```
┌─── CAMERAS (RTSP / USB) ─────────────────────────────────────┐
│  cam 1    cam 2    cam 3    cam 4                             │
└──────┬─────────┬─────────┬─────────┬──────────────────────────┘
       │         │         │         │
       ▼         ▼         ▼         ▼
┌──────────────────────────────────────────────────────────────┐
│  CameraManager — threaded RTSP capture, resize outside lock  │
└───────────────────────┬──────────────────────────────────────┘
                        │ (batch of N frames)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Detector — single GPU batched inference pass                │
│    ├─ YOLOv8-pose    → bboxes + 17 keypoints                 │
│    ├─ PersonTracker  → per-camera centroid tracker           │
│    ├─ FaceAgeEstimator — temporal voting:                    │
│    │     InsightFace match → known person + category         │
│    │     no match → intruder after N failed attempts         │
│    │     fallback → pose + Haar + bbox voting (child/adult)  │
│    └─ FallDetector   → body angle / ratio / hip velocity     │
└───────────────────────┬──────────────────────────────────────┘
                        │ (annotated frame + events)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Alert pipeline                                               │
│    ├─ Safe-zone filter (suppress lying in bed, etc.)         │
│    ├─ Cooldown check                                         │
│    ├─ Snapshot written async to disk                         │
│    ├─ DB event row                                           │
│    ├─ Intruder record + face crop (if intruder_detected)     │
│    └─ Twilio WhatsApp alert                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project structure

```
HomeShield/
├── app.py                   # Flask app, REST API, processing loop
├── detector.py              # YOLOv8-pose + tracker + fall detector + age
├── face_recognizer.py       # InsightFace wrapper + registry matching
├── camera_manager.py        # Threaded multi-camera capture
├── database.py              # SQLite — events, persons, intruders, zones
├── alerter.py               # Twilio WhatsApp with cooldown
├── config.py                # Centralised config (reads .env)
├── requirements.txt         # Pinned dependencies
├── .env.example             # Configuration template
├── templates/
│   └── index.html           # Single-file dashboard (HTML + CSS + JS)
├── snapshots/               # Event photos, face thumbnails (git-ignored)
│   ├── persons/             #   Registered person thumbnails
│   └── intruders/           #   Intruder face crops
└── homeshield.db            # SQLite database (git-ignored)
```

---

## 🔐 Privacy & security

- **No video uploads** — all inference runs locally. Video frames never leave the machine.
- **Face data stays local** — embeddings and thumbnails live in your local SQLite DB and `snapshots/` folder. The `.gitignore` blocks both from being committed.
- **WhatsApp alerts** go through Twilio's API (a text message + optional image URL). Anthropic/HomeShield has no visibility into your Twilio traffic.
- **Treat `homeshield.db` as sensitive** — it contains face embeddings of your family members. Don't email it, don't commit it to Git.

---

## 🧪 Performance notes

On an RTX 4070 with the recommended `yolov8l-pose` at 640 input size:
- Single-frame inference: **~12 ms**
- Four-camera batched inference: **~22 ms** (5.5 ms per camera effective)
- End-to-end latency from fall to WhatsApp send: **< 1 second**

On CPU (Intel i5-12400) with `yolov8n-pose` at 480:
- Single-frame inference: **~90 ms**
- Can sustain ~8 FPS for one camera or ~3 FPS for four cameras
- Fall response latency: **2–3 seconds**

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
Install deps: `pip install -r requirements.txt`. If that fails for `insightface`, see the Windows install section above.

**`numpy.dtype size changed, may indicate binary incompatibility`**
NumPy 2.x is incompatible with the InsightFace wheel. Run `pip install "numpy<2.0"`.

**`torch.cuda.is_available()` is False with an NVIDIA GPU**
You installed the CPU build of PyTorch. Reinstall with the CUDA index URL from the installation section.

**Port 5000 in use** (common on macOS — AirPlay uses it)
Change `PORT=5001` in `.env`, restart, open http://localhost:5001.

**YOLO model change doesn't take effect**
The detector reloads on Stop → Start when the model / image size / FP16 setting changes. If a reload doesn't pick up a change, fully restart the Python process.

**InsightFace takes forever on first run**
First run downloads ~300 MB of model weights to `~/.insightface/models/buffalo_l/`. This only happens once.

**No faces detected in the live preview**
Move closer to the camera. The preview requires the face to be at least 10% of the frame height for a reliable embedding.

---

## 🗺️ Development roadmap

This repository is the FYP1 submission. Planned for FYP2:
- CNN-LSTM action recognition for more nuanced activities (choking, stumbling, reaching for medicine cabinet)
- Multi-person re-identification across cameras (track "Grandma walked from bedroom to kitchen")
- Mobile app companion for event review and live viewing
- Audio anomaly detection (screams, glass breaking) to supplement vision

---

## 📄 License

This project is released for educational and research use as part of a Final Year Project at the International Islamic University Malaysia.

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — person detection and pose
- [InsightFace](https://github.com/deepinsight/insightface) — face recognition, age, and gender estimation
- [Gourieff](https://github.com/Gourieff/Assets) — prebuilt InsightFace Windows wheels
- [Twilio](https://www.twilio.com) — WhatsApp Business API
