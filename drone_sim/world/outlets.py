"""Wall outlet locations and safe inspection viewpoints."""

from dataclasses import dataclass
import numpy as np

from drone_sim.validation import positive


@dataclass(frozen = True)
class PowerOutlet:
    name: str
    position: tuple[float, float, float]
    inward_normal: tuple[float, float, float]

    def inspection_point(self, room, standoff, height, inspection_range):
        standoff = positive(standoff, "Outlet standoff")
        inspection_range = positive(inspection_range, "Inspection range")
        point = np.asarray(self.position) + standoff * np.asarray(self.inward_normal)
        point[2] = float(height)
        if not np.isfinite(point[2]) or not room.base_height < point[2] < room.base_height + room.wall_height:
            raise ValueError("Inspection height must lie inside the room height.")
        if np.linalg.norm(point - self.position) > inspection_range:
            raise ValueError("Inspection point is outside the outlet's 3D inspection range.")
        if not room.segment_xy_is_clear(point[:2], point[:2], room.route_margin):
            raise ValueError("Outlet inspection point overlaps a red zone or route margin.")
        return point

    def facing_yaw(self):
        normal = self.inward_normal
        return float(np.arctan2(-normal[1], -normal[0]))


def create_wall_outlets(room, specifications):
    """Read (wall, coordinate along wall, height above floor) specifications."""
    outlets = []
    for index, (wall, coordinate, height) in enumerate(specifications, start = 1):
        coordinate, height = float(coordinate), float(height)
        if not np.isfinite(coordinate) or not np.isfinite(height):
            raise ValueError("Outlet coordinates must be finite.")
        if not 0.09 < height < room.wall_height - 0.09:
            raise ValueError("Outlet plate must fit within the wall height.")
        x, y = room.width / 2, room.length / 2
        walls = {"west": ((-x, coordinate), (1, 0, 0), y),
                 "east": ((x, coordinate), (-1, 0, 0), y),
                 "south": ((coordinate, -y), (0, 1, 0), x),
                 "north": ((coordinate, y), (0, -1, 0), x)}
        if wall not in walls:
            raise ValueError("Outlet wall must be west, east, south or north.")
        xy, normal, half_span = walls[wall]
        if abs(coordinate) + 0.06 >= half_span:
            raise ValueError("Outlet plate must fit within the wall width.")
        outlets.append(PowerOutlet(f"Outlet {index}",
            (xy[0], xy[1], room.base_height + height), normal))
    return tuple(outlets)
