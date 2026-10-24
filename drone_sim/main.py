"""Entry point for the inspection scaffold."""

from drone_sim.config import AppConfig


def main() -> None:
    config = AppConfig()
    print(f"Inspection scaffold: {config.simulation.scene} scene, PyBullet physics.")
    print("Capture, detection, API analysis and runtime orchestration are pending.")


if __name__ == "__main__":
    main()
