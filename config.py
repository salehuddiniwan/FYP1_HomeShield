"""
HomeShield Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "homeshield-secret-key-change-me")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "homeshield.db")

    # Snapshots
    SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "static/snapshots")

    # Detection
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m.pt")
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.5))
    POSE_MIN_DETECTION_CONFIDENCE = float(os.getenv("POSE_CONFIDENCE", 0.5))
    POSE_MIN_TRACKING_CONFIDENCE = float(os.getenv("POSE_TRACKING", 0.5))

    # ── GPU / Inference settings ──────────────────────────────────────────────
    # Device for YOLO inference: "cuda", "cuda:0", "cpu", or "mps" (Apple Silicon)
    # Set to "auto" to let HomeShield pick the best available device automatically.
    GPU_DEVICE = os.getenv("GPU_DEVICE", "auto")

    # Enable FP16 (half-precision) on GPU for ~2x speed with minimal accuracy loss.
    # Automatically disabled when GPU_DEVICE resolves to "cpu".
    USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

    # Max frames batched together for multi-camera YOLO inference.
    # Higher = better GPU utilisation but slightly more latency.
    # Recommended: number of active cameras (1-8).
    INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", 4))

    # Size frames are resized to before YOLO inference (must be multiple of 32).
    # Smaller = faster inference; 640 is YOLOv8 native size.
    YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", 640))

    # Number of worker threads for parallel pose estimation (MediaPipe, CPU-bound).
    POSE_WORKERS = int(os.getenv("POSE_WORKERS", 4))

    # Pre-warm the model on startup to eliminate first-frame latency.
    WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", "3"))
    # ─────────────────────────────────────────────────────────────────────────

    # Alert thresholds
    FALL_CONFIDENCE_THRESHOLD = float(os.getenv("FALL_THRESHOLD", 0.80))
    INACTIVITY_SECONDS = int(os.getenv("INACTIVITY_SECONDS", 300))
    ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN", 60))

    # Frame processing
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
    PROCESS_FPS = int(os.getenv("PROCESS_FPS", 15))   # bumped 8 -> 15 with GPU

    # Twilio WhatsApp
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+1122334455")
    ALERT_PHONE_NUMBERS = [
        p.strip() for p in os.getenv("ALERT_PHONE_NUMBERS", "").split(",") if p.strip()
    ]

    # Cameras (JSON list or comma-separated RTSP URLs)
    DEFAULT_CAMERAS = [
        {"name": "Camera 1", "url": "0", "location": "Living Room", "active": True},
    ]
