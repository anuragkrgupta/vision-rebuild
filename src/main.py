"""
Week 1 goal: real-time detection loop with a visible bounding-box overlay
and an on-screen FPS counter, so you can see the pipeline actually working
end to end before adding distance estimation, audio, or logging.

Run from the project root:
    python -m src.main
"""

import time

import cv2

from .camera import Camera
from .detector import Detector


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


def draw_fps(frame, fps: float):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )
    return frame


def main():
    detector = Detector()

    with Camera() as cam:
        prev_time = time.time()

        print("VISION running. Press 'q' in the video window to quit.")

        while True:
            success, frame = cam.read_frame()
            if not success:
                print("Failed to read frame from camera — stopping.")
                break

            detections = detector.detect(frame)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            frame = draw_detections(frame, detections)
            frame = draw_fps(frame, fps)

            cv2.imshow("VISION - Week 1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
