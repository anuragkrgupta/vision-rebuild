# VISION — Real-Time Object Detection for Accessibility

A real-time object detection system built to assist visually impaired users
by converting camera input into audio feedback about nearby objects and
their approximate distance.

This rebuild moves from the original TensorFlow.js/COCO-SSD version to a
Python-based pipeline (YOLOv8 + OpenCV) for better accuracy, real evaluation
tooling, and room to fine-tune on custom data later.

## Status: Week 3 — Logging + Evaluation

Currently implemented:
- Real-time webcam capture (`src/camera.py`)
- YOLOv8 inference wrapper (`src/detector.py`)
- Live bounding-box + FPS overlay (`src/main.py`)
- Monocular distance estimation via known-height heuristic (`src/distance.py`)
- Priority-based text-to-speech feedback with cooldowns (`src/audio.py`)
- Detection logging to CSV or SQLite, on a background thread so it never
  costs a video frame (`src/logger.py`)
- Per-class precision/recall/F1 evaluation against a labeled test set
  (`src/evaluate.py`)

Not yet implemented (upcoming weeks):
- Fine-tuning on custom/domain-specific data
- Config-driven class filtering presets for common scenarios (indoor nav,
  outdoor/street crossing, etc.)

### Detection logging

Every run of `python -m src.main` logs detections to
`data/logs/detections.csv` (or a SQLite DB if you set `LOG_FORMAT =
"sqlite"` in `config.py`). Each row has a timestamp, frame number, class,
confidence, estimated distance, and bounding box — enough to reconstruct
class frequency and distance distributions after a session. Toggle it off
entirely with `LOGGING_ENABLED = False`.

### Evaluation

`src/evaluate.py` measures precision, recall, and F1 per class by running
the actual `Detector` (your real confidence/IOU thresholds and class
filter) against a labeled test set and comparing to ground truth via IOU
matching. Expected layout:

```
data/test_set/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt      # YOLO format: class_id x_center y_center width height (normalized)
    └── img002.txt
```

Run it with:

```bash
python -m src.evaluate
```

This prints a per-class table to the console and writes a timestamped CSV
report to `data/logs/`.

### A note on distance accuracy

`distance.py` uses a heuristic (known average real-world object height vs.
pixel height in frame), not true depth sensing. It's explainable and good
enough for a working demo, but it's not lab-accurate — it assumes the
default `FOCAL_LENGTH_PX` in `config.py`, which is a reasonable guess for a
typical laptop webcam, not your specific one. For meaningfully better
accuracy, calibrate it for your camera using `distance.calibrate()` (see
the docstring in that file for how).

### A note on audio

`pyttsx3` uses your OS's native TTS engine. On Linux this usually needs
`espeak` installed (`sudo apt install espeak`); on Windows/Mac it uses the
built-in voices with no extra setup.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

First run will auto-download `yolov8n.pt` (~6MB) into your working directory
via Ultralytics — no manual model download needed.

## Run

```bash
python -m src.main
```

Press `q` in the video window to quit.

## Tests

```bash
pip install pytest
pytest tests/
```

## Project Structure

```
vision/
├── src/
│   ├── config.py     # all tunables — thresholds, model, camera, logging, eval settings
│   ├── detector.py    # YOLOv8 wrapper
│   ├── camera.py       # webcam capture
│   ├── distance.py       # monocular distance estimation heuristic
│   ├── audio.py             # priority-based TTS feedback
│   ├── logger.py               # detection logging (CSV/SQLite), background-threaded
│   ├── evaluate.py                # per-class precision/recall/F1 on a labeled test set
│   └── main.py                       # ties it together, live display
├── models/                 # downloaded/fine-tuned weights (gitignored)
├── data/
│   ├── logs/                # detection logs + eval reports (gitignored)
│   └── test_set/               # images/ + labels/ for evaluate.py (not committed)
└── tests/
    ├── test_detector.py
    ├── test_logger.py
    └── test_evaluate.py
```

## Design notes

- `detector.py` and `camera.py` are deliberately decoupled — the model or
  the video source can be swapped independently.
- All thresholds live in `config.py` so tuning doesn't mean hunting through
  logic code.
- `yolov8n.pt` (nano) is the starting model: fastest, lowest accuracy of the
  YOLOv8 family. Swap to `yolov8s.pt` in `config.py` if accuracy matters more
  than FPS for your hardware.
