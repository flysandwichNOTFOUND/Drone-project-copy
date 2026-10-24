"""Drone model and control interface backed by PyBullet."""

from drone_sim.simulation.world import PyBulletWorld


class InspectionDrone:
    def __init__(self, world: PyBulletWorld):
        self.world = world
        self.body_id: int | None = None

    def load(self, urdf_path: str) -> None:
        """Load the selected drone asset into the connected world."""
        raise NotImplementedError("Drone asset setup is pending.")

    def apply_motor_commands(self, thrusts: tuple[float, float, float, float]) -> None:
        """Apply motor forces and torques through PyBullet."""
        raise NotImplementedError("Motor control is pending.")

    def get_state(self) -> dict:
        """Read pose and velocity from PyBullet."""
        raise NotImplementedError("Drone state access is pending.")
