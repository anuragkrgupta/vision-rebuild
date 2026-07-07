"""
Central config for VISION.
Keep every tunable here so tuning/debugging never means hunting through
detector.py or camera.py.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

# --- Model ---
# yolov8n = nano: fastest, lowest accuracy. Good starting point for real-time
# on modest hardware. Swap to yolov8s.pt later if accuracy matters more than FPS.
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# --- Detection thresholds ---
CONFIDENCE_THRESHOLD = 0.45   # below this, we don't trust the detection enough to act on it
IOU_THRESHOLD = 0.45          # non-max suppression overlap threshold

# --- Camera ---
CAMERA_INDEX = 0              # 0 = default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Classes we care about for the assistive-tech use case ---
# None = detect everything YOLO knows. Restrict later once you decide which
# object classes actually matter for navigation (person, chair, car, etc.)
TARGET_CLASSES = None
