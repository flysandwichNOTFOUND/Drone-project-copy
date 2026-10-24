"""Records passed between capture, detection and analysis."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class CameraFrame:
    frame_id: int
    captured_at: float
    image: Any  # OpenCV-compatible image array.


@dataclass
class Detection:
    label: Literal["wire", "plug"]
    confidence: float
    bounding_box: tuple[int, int, int, int]  # x, y, width, height in pixels.


@dataclass
class InspectionEvidence:
    frame_id: int
    captured_at: float
    scene: Literal["indoor", "outdoor"]
    detections: list[Detection]
    observations: tuple[str, ...] = ()


@dataclass
class HazardAssessment:
    frame_id: int
    risk_level: Literal["unknown", "low", "medium", "high"]
    explanation: str
    limitations: tuple[str, ...] = ()
