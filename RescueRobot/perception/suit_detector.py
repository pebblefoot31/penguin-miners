"""
perception/suit_detector.py
===========================

YOLO-based detector for people wearing the predefined rescue suit. People who
are *not* wearing the suit are ignored. A secondary HSV color gate (from
``config/rescue_suit.yaml``) suppresses false positives so only genuine
rescue-suit wearers are returned.

Single responsibility: turn an RGB frame into a list of validated rescue-suit
detections.
"""

from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np

from utils.logger import get_logger
from utils.types import BoundingBox, Detection


class SuitDetector:
    """Detects rescue-suit wearers using YOLO + a color/pattern gate."""

    def __init__(
        self,
        weights: str,
        target_class_name: str,
        confidence_threshold: float,
        iou_threshold: float,
        device: str,
        suit_signature: dict,
        mock: bool = False,
    ) -> None:
        self._log = get_logger("perception.suit_detector")
        self._weights = weights
        self._target_class = target_class_name
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._device = device
        self._signature = suit_signature or {}
        self._mock = mock
        self._model: Optional[Any] = None
        if not mock:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self._weights)
            self._log.info("YOLO weights loaded from %s.", self._weights)
        except Exception as exc:
            self._log.warning(
                "YOLO unavailable (%s); using mock color-blob detector.", exc
            )
            self._mock = True

    def detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        """Return validated rescue-suit detections for the given RGB frame."""
        raw = self._run_model(rgb_frame)
        validated: List[Detection] = []
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        for det in raw:
            det.color_match_fraction = self._color_match_fraction(bgr, det.bbox)
            det.is_rescue_suit = self._is_rescue_suit(det)
            if det.is_rescue_suit:
                validated.append(det)
        self._log.debug(
            "Detections: %d raw, %d validated rescue-suit.", len(raw), len(validated)
        )
        return validated

    # --------------------------------------------------------------- internals
    def _run_model(self, rgb_frame: np.ndarray) -> List[Detection]:
        """Run YOLO (or the mock detector) and return unvalidated detections."""
        if self._mock or self._model is None:
            return self._mock_detect(rgb_frame)

        results = self._model.predict(
            source=rgb_frame,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id))
                if class_name != self._target_class:
                    continue  # ignore non-suit classes (e.g. plain "person")
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        ),
                        confidence=float(box.conf[0]),
                        class_name=class_name,
                    )
                )
        return detections

    def _mock_detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        """Fallback detector: find the largest suit-colored blob."""
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        mask = self._suit_color_mask(bgr)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 400:
            return []
        x, y, w, h = cv2.boundingRect(largest)
        return [
            Detection(
                bbox=BoundingBox(x, y, x + w, y + h),
                confidence=0.8,
                class_name=self._target_class,
            )
        ]

    def _suit_color_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Binary mask of pixels matching the rescue-suit HSV color gate."""
        gate = self._signature.get("color_gate", {})
        lower = np.array(gate.get("hsv_lower", [5, 120, 90]), dtype=np.uint8)
        upper = np.array(gate.get("hsv_upper", [22, 255, 255]), dtype=np.uint8)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, lower, upper)

    def _color_match_fraction(
        self, bgr_frame: np.ndarray, bbox: BoundingBox
    ) -> float:
        """Fraction of bbox pixels matching the suit color (0..1)."""
        crop = bgr_frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if crop.size == 0:
            return 0.0
        mask = self._suit_color_mask(crop)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _is_rescue_suit(self, detection: Detection) -> bool:
        """Apply the color gate to confirm the detection is a real suit."""
        gate = self._signature.get("color_gate", {})
        if not gate.get("enabled", True):
            return detection.confidence >= self._conf
        min_fraction = float(gate.get("min_color_fraction", 0.18))
        return (
            detection.confidence >= self._conf
            and detection.color_match_fraction >= min_fraction
        )
