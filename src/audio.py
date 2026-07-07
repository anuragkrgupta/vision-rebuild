"""
Audio feedback with priority logic.

Two problems a naive "announce every detection every frame" approach has:
1. It's a wall of noise — 20+ announcements per second, useless to a user.
2. TTS engines block while speaking — calling speak() every frame freezes
   the video loop.

This module fixes both: runs TTS on a background thread via a queue, and
only announces the closest N objects, with a per-class cooldown so the same
object isn't re-announced every frame while it sits in view.
"""

import queue
import threading
import time
from typing import List, Optional, Tuple

import pyttsx3

from .detector import Detection


class AudioFeedback:
    def __init__(
        self,
        max_announcements_per_cycle: int = 2,
        cooldown_seconds: float = 4.0,
        speech_rate: int = 175,
    ):
        self.max_announcements_per_cycle = max_announcements_per_cycle
        self.cooldown_seconds = cooldown_seconds

        self._last_announced: dict[str, float] = {}  # class_name -> last spoken time
        self._speech_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_flag = threading.Event()

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", speech_rate)

        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

    def _speech_worker(self):
        """Runs on a background thread so speaking never blocks the video loop."""
        while not self._stop_flag.is_set():
            try:
                text = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._engine.say(text)
            self._engine.runAndWait()

    def announce(self, detections_with_distance: List[Tuple[Detection, Optional[float]]]):
        """
        Given detections paired with their estimated distance (meters, or
        None if unknown), decide what — if anything — to say this cycle.
        """
        # Only consider objects we could estimate a distance for, closest first.
        known_distance = [
            (det, dist) for det, dist in detections_with_distance if dist is not None
        ]
        known_distance.sort(key=lambda pair: pair[1])

        now = time.time()
        announced_this_cycle = 0

        for det, dist in known_distance:
            if announced_this_cycle >= self.max_announcements_per_cycle:
                break

            last_time = self._last_announced.get(det.class_name, 0.0)
            if now - last_time < self.cooldown_seconds:
                continue  # said this recently, don't repeat yet

            phrase = self._build_phrase(det.class_name, dist)
            self._speech_queue.put(phrase)
            self._last_announced[det.class_name] = now
            announced_this_cycle += 1

    @staticmethod
    def _build_phrase(class_name: str, distance_m: float) -> str:
        if distance_m < 1.0:
            urgency = "very close"
        elif distance_m < 2.5:
            urgency = "nearby"
        else:
            urgency = "ahead"
        return f"{class_name} {urgency}, {distance_m:.1f} meters"

    def stop(self):
        self._stop_flag.set()
        self._worker.join(timeout=1.0)
