#!/bin/bash
# HomeShield — One-click setup script
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║         🛡️  HomeShield Setup                  ║"
echo "║  ML-Powered CCTV Surveillance System         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is required. Install from https://python.org"
    exit 1
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python version: $PYVER"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[INFO] Created .env file — please edit with your settings"
fi

# Create directories
mkdir -p static/snapshots
mkdir -p models

# Download YOLOv8 model if not present
if [ ! -f "yolov8n.pt" ]; then
    echo "[INFO] Downloading YOLOv8n model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  1. Edit .env with your Twilio credentials"
echo "  2. Run: python app.py"
echo "  3. Open: http://localhost:5000"
echo "═══════════════════════════════════════════════"
