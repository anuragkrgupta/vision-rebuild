"""
Webcam capture. Kept separate from detection logic so you can later swap
in a video file, an RTSP stream, or a phone camera feed without touching
detector.py or main.py.
"""

import cv2

from . import config


class Camera:
    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {camera_index}. "
                "Check the index, or that no other app is using the camera."
            )

    def read_frame(self):
        """Returns (success, frame). Caller should check success before use."""
        return self.cap.read()

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
