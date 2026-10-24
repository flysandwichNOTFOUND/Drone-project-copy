import numpy as np
from drone_sim.validation import positive, vector


class FlatTerrain:
    def __init__(self, base_height = 0.0):

        self.base_height = float(base_height)

        if not np.isfinite(self.base_height):
            raise ValueError("Terrain base height must be finite.")

    def height(self, x, y):

        x, y = np.broadcast_arrays(x, y)

        return np.full(x.shape, self.base_height)

    def gradient(self, x, y):

        return np.zeros(2)

    def normal(self, x, y):

        gradient = self.gradient(x, y)
        normal = np.r_[-gradient, 1.0]

        return normal / np.linalg.norm(normal)

    def maximum_height_along(self, start, goal):

        return self.base_height


class HillTerrain(FlatTerrain):
    def __init__(self, base_height = 0.0, center = (5.0, 4.0), height = 3.0, width = 1.6):

        super().__init__(base_height)
        self.center = vector(center, 2, "Hill center")
        self.hill_height = positive(height, "Hill height", allow_zero = True)
        self.width = positive(width, "Hill width")

    def height(self, x, y):

        return self.base_height + self.hill_height * np.exp( -((np.asarray(x) - self.center[0])**2 + (np.asarray(y) - self.center[1])**2) / (2 * self.width**2))

    def gradient(self, x, y):

        return -(float(self.height(x, y)) - self.base_height) * ( np.array([x, y]) - self.center) / self.width**2

    def maximum_height_along(self, start, goal):

        start, goal = np.asarray(start), np.asarray(goal)
        direction = goal - start
        length_squared = np.dot(direction, direction)
        fraction = (np.clip(np.dot(self.center - start, direction) / length_squared, 0, 1)
                    if length_squared else 0.0)
        closest = start + fraction * direction

        return float(self.height(*closest))


class SlopedTerrain(FlatTerrain):
    def __init__(self, base_height = 0.0, gradient = (0.1, 0.0)):

        super().__init__(base_height)
        self.slope = vector(gradient, 2, "Terrain gradient")

    def height(self, x, y):

        return self.base_height + self.slope[0] * np.asarray(x) + self.slope[1] * np.asarray(y)

    def gradient(self, x, y):

        return self.slope.copy()

    def maximum_height_along(self, start, goal):

        return max(float(self.height(*start)), float(self.height(*goal)))


class RoomTerrain(FlatTerrain):
    """Flat floor with a rectangular, open-top enclosure.

    Width/length measure the clear interior; wall thickness extends outward.
    Floor elevation stays separate from the vertical wall collision geometry.
    """

    def __init__(self, base_height = 0.0, width = 20.0, length = 20.0,
                 wall_height = 4.0, wall_thickness = 0.2,
                 pillar_center = (-2.5, 2.5), pillar_width = 0.5, pillar_height = 4.0,
                 minimum_safe_distance = 1.0, route_margin = 0.1, outlet_specs = ()):
        super().__init__(base_height)
        self.width = positive(width, "Room width")
        self.length = positive(length, "Room length")
        self.wall_height = positive(wall_height, "Wall height")
        self.wall_thickness = positive(wall_thickness, "Wall thickness")
        self.plot_bounds = (-self.width / 2, self.width / 2,
                            -self.length / 2, self.length / 2)
        self.pillar_center = vector(pillar_center, 2, "Pillar centre")
        self.pillar_width = positive(pillar_width, "Pillar width")
        self.pillar_height = positive(pillar_height, "Pillar height")
        self.minimum_safe_distance = positive(minimum_safe_distance, "Minimum safe distance")
        self.route_margin = positive(route_margin, "Route margin", allow_zero = True)
        if np.any(np.abs(self.pillar_center) + self.pillar_width / 2
                  >= np.array([self.width, self.length]) / 2):
            raise ValueError("Pillar must fit inside the room.")
        from drone_sim.world.outlets import create_wall_outlets
        self.outlets = create_wall_outlets(self, outlet_specs)

    def blocked_pillar_bounds(self, extra_clearance = 0.0):
        """Pillar footprints expanded by the red zone and optional route margin."""
        clearance = self.minimum_safe_distance + positive(
            extra_clearance, "Extra clearance", allow_zero = True)
        return [(x - hx - clearance, x + hx + clearance,
                 y - hy - clearance, y + hy + clearance)
                for (x, y, _), (hx, hy, _) in self.pillar_boxes()]

    def segment_xy_is_clear(self, start_xy, end_xy, extra_clearance = 0.0):
        """Certify the entire segment stays outside closed red rectangles.

        Walls bound a convex safe interior. Slab intersection tests reject
        pillar-zone crossings, corner cutting and even boundary tangency.
        These are horizontal no-go zones; climbing never bypasses them.
        """
        start = vector(start_xy, 2, "Segment start XY")
        end = vector(end_xy, 2, "Segment end XY")
        clearance = self.minimum_safe_distance + positive(
            extra_clearance, "Extra clearance", allow_zero = True)
        xmin, xmax, ymin, ymax = self.plot_bounds
        for point in (start, end):
            if not (xmin + clearance < point[0] < xmax - clearance
                    and ymin + clearance < point[1] < ymax - clearance):
                return False
        direction = end - start
        for bounds in self.blocked_pillar_bounds(extra_clearance):
            enter, leave = 0.0, 1.0
            for axis, (lower, upper) in enumerate(((bounds[0], bounds[1]),
                                                    (bounds[2], bounds[3]))):
                if abs(direction[axis]) < 1e-14:
                    if not lower <= start[axis] <= upper:
                        break
                else:
                    first, last = sorted(((lower - start[axis]) / direction[axis],
                                          (upper - start[axis]) / direction[axis]))
                    enter, leave = max(enter, first), min(leave, last)
                    if enter > leave:
                        break
            else:
                return False  # Both axis intervals intersect the segment.
        return True

    def pillar_boxes(self):
        """Return the pillar centre and half-extents for physics and plots."""
        x, y = self.pillar_center
        half_width, half_height = self.pillar_width / 2, self.pillar_height / 2
        return (((x, y, self.base_height + half_height),
                 (half_width, half_width, half_height)),)

    def wall_boxes(self):
        """Return (centre, half-extents) pairs for the four solid walls."""
        x, y = self.width / 2, self.length / 2
        t, h = self.wall_thickness / 2, self.wall_height / 2
        z = self.base_height + h
        return (
            ((-x - t, 0, z), (t, y + 2 * t, h)),
            ((x + t, 0, z), (t, y + 2 * t, h)),
            ((0, -y - t, z), (x, t, h)),
            ((0, y + t, z), (x, t, h)),
        )
