"""
YOLOv8 detection wrapper.
Isolated from camera/display code so the model can be swapped or fine-tuned
later without touching anything else.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    """One detected object in a single frame."""
    class_name: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) pixel coordinates
    box_height_px: int


class Detector:
    def __init__(
        self,
        model_path: str = str(config.MODEL_PATH),
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        target_classes: Optional[List[str]] = config.TARGET_CLASSES,
    ):
        # Ultralytics auto-downloads the base weights on first run if not
        # present locally at model_path.
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.class_names = self.model.names  # id -> name mapping from the model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame, return filtered detections."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_height_px = int(y2 - y1)

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    box_height_px=box_height_px,
                )
            )

        return detections
