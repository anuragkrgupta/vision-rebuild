"""
Tests for the detection logger — CSV and SQLite backends, plus the
disabled no-op path. Run with: pytest tests/
"""

import csv
import sqlite3
import tempfile
from pathlib import Path

from src.detector import Detection
from src.logger import DetectionLogger


def _sample_detection():
    return Detection(class_name="person", confidence=0.91, box_xyxy=(10, 20, 110, 220), box_height_px=200)


def test_csv_logging_writes_header_and_row():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "detections.csv"
        logger = DetectionLogger(log_path=log_path, log_format="csv", enabled=True)

        logger.log([(_sample_detection(), 2.5)])
        logger.stop()

        with open(log_path, newline="") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "timestamp", "frame_id", "class_name", "confidence",
            "distance_m", "x1", "y1", "x2", "y2",
        ]
        assert len(rows) == 2
        assert rows[1][2] == "person"


def test_csv_logging_appends_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "detections.csv"

        logger1 = DetectionLogger(log_path=log_path, log_format="csv", enabled=True)
        logger1.log([(_sample_detection(), 1.0)])
        logger1.stop()

        logger2 = DetectionLogger(log_path=log_path, log_format="csv", enabled=True)
        logger2.log([(_sample_detection(), 1.5)])
        logger2.stop()

        with open(log_path, newline="") as f:
            rows = list(csv.reader(f))

        # One header + two data rows, header not duplicated on the second run.
        assert len(rows) == 3


def test_sqlite_logging_creates_table_and_row():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "detections.db"
        logger = DetectionLogger(log_path=log_path, log_format="sqlite", enabled=True)

        logger.log([(_sample_detection(), 1.2)])
        logger.stop()

        conn = sqlite3.connect(log_path)
        rows = conn.execute("SELECT class_name, confidence FROM detections").fetchall()
        conn.close()

        assert rows == [("person", 0.91)]


def test_disabled_logger_is_a_no_op():
    missing_path = Path(tempfile.gettempdir()) / "vision_test_should_not_exist.csv"
    if missing_path.exists():
        missing_path.unlink()

    logger = DetectionLogger(log_path=missing_path, enabled=False)
    logger.log([(_sample_detection(), 2.5)])  # should not raise
    logger.stop()  # should not raise

    assert not missing_path.exists()


def test_invalid_log_format_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            DetectionLogger(log_path=Path(tmp) / "x.csv", log_format="xml", enabled=True)
            assert False, "expected ValueError for unknown log_format"
        except ValueError:
            pass
