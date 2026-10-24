"""Collision avoidance interface; controller integration will follow later."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from drone_sim.physics.dynamics import DroneState
from drone_sim.validation import positive


class CollisionCheck:
    """Track multiple walls/objects and request protective controller actions.

    Distances have shape (N,) in metres. Directions have shape (N, 3),
    with each unit vector pointing FROM the drone TOWARD the corresponding
    wall/object in world coordinates. Rows must refer to the same objects.
    An empty scene uses shapes (0,) and (0, 3).

    Controller calls are intentionally deferred until its avoidance interface
    is defined. This scaffold does not yet provide collision protection.
    """

    SLOWDOWN_THRESHOLD = 0.5  # Metres outside the minimum safe distance.

    def __init__(self, controller, wall_distances: ArrayLike,
                 wall_directions: ArrayLike, minimum_safe_distance: float):

        self.controller = controller
        self.minimum_safe_distance = positive(
            minimum_safe_distance, "Minimum safe distance")
        self.slowdown_distance = self.minimum_safe_distance + self.SLOWDOWN_THRESHOLD
        self.wall_distances = np.asarray(wall_distances, dtype = float).copy()
        self.wall_directions = np.asarray(wall_directions, dtype = float).copy()
        self._validate_wall_vectors()

    def _validate_wall_vectors(self) -> None:
        """Require paired finite distances and world-space unit directions."""

        if self.wall_distances.ndim != 1:
            raise ValueError("Wall distances must have shape (N,).")
        if self.wall_directions.shape != (self.wall_distances.size, 3):
            raise ValueError("Wall directions must have shape (N, 3).")
        if not np.all(np.isfinite(self.wall_distances)) or np.any(self.wall_distances < 0):
            raise ValueError("Wall distances must be finite and nonnegative.")
        if not np.all(np.isfinite(self.wall_directions)) or not np.allclose(
                np.linalg.norm(self.wall_directions, axis = 1), 1.0):
            raise ValueError("Wall directions must be finite unit vectors.")

    def update_walls(self, wall_distances: ArrayLike, wall_directions: ArrayLike) -> None:
        """Replace paired distance/direction readings after validating them."""
        raise NotImplementedError("Live obstacle updates will be implemented next.")

    def check_safety_zones(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return per-wall (unsafe, slowdown) masks.

        Unsafe: distance <= minimum_safe_distance.
        Slowdown: minimum_safe_distance < distance <= slowdown_distance.
        Beyond slowdown_distance: no restriction from that wall.
        """
        raise NotImplementedError("Safety-zone classification is pending.")

    def slow_down(self, state: DroneState, dt: float) -> None:
        """Ask the controller to reduce motion toward all slowdown-zone walls.

        Prepare to stop before the minimum safe distance, accounting for
        current velocity and simultaneous constraints from multiple walls.
        """
        raise NotImplementedError("Controller braking integration is pending.")

    def stop_and_move_away(self, state: DroneState, dt: float) -> None:
        """Ask the controller to stop approach and recover safe clearance.

        Resolve all nearby wall constraints together; do not correct away
        from one wall by commanding motion into another.
        """
        raise NotImplementedError("Controller clearance recovery is pending.")

    def enforce_safety(self, state: DroneState, dt: float) -> None:
        """Check zones and dispatch braking/recovery with unsafe zones first."""
        raise NotImplementedError("Collision avoidance orchestration is pending.")
