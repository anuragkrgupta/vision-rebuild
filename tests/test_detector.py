"""
Basic sanity tests — enough to catch obvious breakage, not a full test suite.
Run with: pytest tests/
"""

import numpy as np

from src.detector import Detector


def test_detector_loads():
    """Model should load without throwing."""
    detector = Detector()
    assert detector.model is not None
    assert len(detector.class_names) > 0


def test_detect_on_blank_frame():
    """A blank frame should run without error and return a list (possibly empty)."""
    detector = Detector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(blank_frame)

    assert isinstance(detections, list)
