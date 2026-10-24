"""Simulated camera interface."""

from drone_sim.config import CameraConfig
from drone_sim.models import CameraFrame
from drone_sim.simulation.drone import InspectionDrone


class DroneCamera:
    def __init__(self, drone: InspectionDrone, config: CameraConfig):
        self.drone = drone
        self.config = config

    def capture(self) -> CameraFrame:
        """Capture a PyBullet camera frame for OpenCV processing."""
        raise NotImplementedError("Camera mounting and capture are pending.")
