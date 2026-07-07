"""
Monocular distance estimation using a known-height heuristic.

The idea: for an object of known real-world height H (meters), if it appears
h pixels tall in the frame, and the camera's focal length is f (pixels), then
distance D = (H * f) / h.

This is a heuristic, not true depth sensing — it assumes the object is
roughly the "average" size for its class and that it's upright and fully
visible. It's explainable and good enough for a working assistive-tech demo.
For real depth, MiDaS (monocular depth estimation model) is the upgrade path.
"""

from typing import Optional

from .detector import Detection

# Average real-world heights (meters) for common COCO classes relevant to
# navigation. Extend this as you test against more object types.
KNOWN_HEIGHTS_M = {
    "person": 1.7,
    "chair": 0.9,
    "couch": 0.85,
    "dining table": 0.75,
    "bottle": 0.25,
    "cup": 0.12,
    "laptop": 0.02,
    "tv": 0.5,
    "car": 1.5,
    "bicycle": 1.0,
    "motorcycle": 1.2,
    "dog": 0.5,
    "backpack": 0.45,
    "suitcase": 0.6,
    "door": 2.0,
}

# Needs one-time calibration for your specific camera (see calibrate() below).
# 615 is a reasonable default for a typical laptop webcam at 640x480.
DEFAULT_FOCAL_LENGTH_PX = 950.0


def estimate_distance(
    detection: Detection,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
) -> Optional[float]:
    """
    Returns estimated distance in meters, or None if we don't have a known
    height for this object class (rather than silently guessing wrong).
    """
    real_height_m = KNOWN_HEIGHTS_M.get(detection.class_name)
    if real_height_m is None or detection.box_height_px <= 0:
        return None

    distance_m = (real_height_m * focal_length_px) / detection.box_height_px
    return round(distance_m, 2)


def calibrate(known_distance_m: float, known_height_m: float, box_height_px: int) -> float:
    """
    One-time calibration helper: place an object of known height at a known
    distance from your camera, measure its bounding box height in pixels,
    then run this to get YOUR camera's real focal length.

    Example: a 1.7m-tall person standing 2m from the camera measures 340px
    tall in frame -> focal_length_px = calibrate(2.0, 1.7, 340)
    Put the result in config.py / pass it into estimate_distance().
    """
    return (box_height_px * known_distance_m) / known_height_m
