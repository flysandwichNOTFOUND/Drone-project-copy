"""Workflow interface for capture, detection and asynchronous assessment."""

from drone_sim.analysis.deepseek import DeepSeekAnalyzer
from drone_sim.models import HazardAssessment
from drone_sim.vision.camera import DroneCamera
from drone_sim.vision.detector import WirePlugDetector


class InspectionPipeline:
    def __init__(self, camera: DroneCamera, detector: WirePlugDetector,
                 analyzer: DeepSeekAnalyzer):
        self.camera = camera
        self.detector = detector
        self.analyzer = analyzer

    async def inspect_once(self) -> HazardAssessment:
        """Capture, detect, build evidence and request an assessment."""
        raise NotImplementedError("Inspection orchestration is pending.")

    async def run(self) -> None:
        """Schedule bounded API work alongside simulation stepping."""
        raise NotImplementedError("Real-time scheduling is pending.")
