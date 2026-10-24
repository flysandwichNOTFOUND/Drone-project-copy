"""Minimal PyBullet lifecycle; PyBullet advances all physical state."""

from drone_sim.config import SimulationConfig


class PyBulletWorld:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client_id: int | None = None

    def connect(self) -> None:
        import pybullet as p

        if self.client_id is not None:
            return
        if self.config.timestep <= 0:
            raise ValueError("Simulation timestep must be positive.")
        client_id = p.connect(p.GUI if self.config.gui else p.DIRECT)
        if client_id < 0:
            raise RuntimeError("Could not connect to PyBullet.")
        self.client_id = client_id
        p.setGravity(0, 0, self.config.gravity, physicsClientId=client_id)
        p.setTimeStep(self.config.timestep, physicsClientId=client_id)

    def step(self) -> None:
        import pybullet as p

        if self.client_id is None:
            raise RuntimeError("Connect the world before stepping.")
        p.stepSimulation(physicsClientId=self.client_id)

    def close(self) -> None:
        import pybullet as p

        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None
