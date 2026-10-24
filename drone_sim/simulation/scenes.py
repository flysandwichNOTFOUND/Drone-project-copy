"""Indoor/outdoor scene construction interfaces."""

from drone_sim.simulation.world import PyBulletWorld


class InspectionScene:
    def build(self, world: PyBulletWorld) -> None:
        raise NotImplementedError("Scene assets are pending.")


class IndoorScene(InspectionScene):
    """Room, sockets, plugs and wires for indoor inspection."""


class OutdoorScene(InspectionScene):
    """Outdoor electrical fixtures and wires for inspection."""


def create_scene(kind: str) -> InspectionScene:
    if kind == "indoor":
        return IndoorScene()
    if kind == "outdoor":
        return OutdoorScene()
    raise ValueError("Scene must be indoor or outdoor.")
