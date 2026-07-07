"""
Week 2 goal: add distance estimation and priority-based audio feedback on
top of the Week 1 detection pipeline. The video window still shows boxes,
labels, and now estimated distance — audio announces only the closest
objects, with a cooldown so it doesn't spam the same object every frame.

Run from the project root:
    python -m src.main
"""

import time

import cv2

from . import config
from .audio import AudioFeedback
from .camera import Camera
from .detector import Detector
from .distance import estimate_distance


def draw_detections(frame, detections_with_distance):
    for det, dist in detections_with_distance:
        x1, y1, x2, y2 = det.box_xyxy
        dist_label = f"{dist:.1f}m" if dist is not None else "?"
        label = f"{det.class_name} {det.confidence:.2f} | {dist_label}"

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
    audio = AudioFeedback(
        max_announcements_per_cycle=config.MAX_ANNOUNCEMENTS_PER_CYCLE,
        cooldown_seconds=config.AUDIO_COOLDOWN_SECONDS,
        speech_rate=config.SPEECH_RATE,
    )

    with Camera() as cam:
        prev_time = time.time()

        print("VISION running. Press 'q' in the video window to quit.")

        try:
            while True:
                success, frame = cam.read_frame()
                if not success:
                    print("Failed to read frame from camera — stopping.")
                    break

                detections = detector.detect(frame)
                detections_with_distance = [
                    (det, estimate_distance(det, config.FOCAL_LENGTH_PX))
                    for det in detections
                ]

                audio.announce(detections_with_distance)

                now = time.time()
                fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
                prev_time = now

                frame = draw_detections(frame, detections_with_distance)
                frame = draw_fps(frame, fps)

                cv2.imshow("VISION - Week 2", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            audio.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
