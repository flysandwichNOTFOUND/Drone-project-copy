"""OpenCV wire and plug detection interface."""

from drone_sim.models import CameraFrame, Detection


class WirePlugDetector:
    def preprocess(self, frame: CameraFrame):
        """Prepare image data using cv2 once the approach is chosen."""
        raise NotImplementedError("OpenCV preprocessing is pending.")

    def detect(self, frame: CameraFrame) -> list[Detection]:
        """Locate wires and plugs with a future detector/model."""
        raise NotImplementedError("Wire and plug detection is pending.")
