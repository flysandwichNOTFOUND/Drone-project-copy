"""Multi-wall velocity constraints and commands for the existing PID trio."""

from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from drone_sim.physics.dynamics import DroneState
from drone_sim.validation import positive, vector


class CollisionCheck:
    """Track multiple walls/objects and request protective controller actions.

    Distances have shape (N,) in metres. Directions have shape (N, 3),
    with each unit vector pointing FROM the drone TOWARD the corresponding
    wall/object in world coordinates. Rows must refer to the same objects.
    An empty scene uses shapes (0,) and (0, 3).

    ``controller`` is the (height, orientation, position) tuple returned by
    main.create_controllers(). Call enforce_safety after the normal mission
    command and before advancing physics; it overrides motor commands only
    when avoidance is needed. This class does not advance or teleport physics.

    Distances must be clearance from the drone's collision envelope, not its
    centre. Braking limits assume the configured deceleration is achievable;
    sensing delay and insufficient braking authority can still cause contact.
    """

    SLOWDOWN_THRESHOLD = 0.5  # Metres outside the minimum safe distance.

    def __init__(self, controller, wall_distances: ArrayLike,
                 wall_directions: ArrayLike, minimum_safe_distance: float,
                 braking_deceleration: float = 0.8, recovery_speed: float = 0.5,
                 response_time: float = 1.0):

        self.controller = controller
        self.minimum_safe_distance = positive(
            minimum_safe_distance, "Minimum safe distance")
        self.slowdown_distance = self.minimum_safe_distance + self.SLOWDOWN_THRESHOLD
        self.braking_deceleration = positive(braking_deceleration, "Braking deceleration")
        self.recovery_speed = positive(recovery_speed, "Recovery speed")
        self.response_time = positive(response_time, "Response time")
        self.recovery_blocked = False
        self.last_safe_velocity = None
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
        previous = self.wall_distances, self.wall_directions
        self.wall_distances = np.asarray(wall_distances, dtype = float).copy()
        try:
            self.wall_directions = np.asarray(wall_directions, dtype = float).copy()
            self._validate_wall_vectors()
        except (TypeError, ValueError):
            self.wall_distances, self.wall_directions = previous
            raise

    def check_safety_zones(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return per-wall (unsafe, slowdown) masks.

        Unsafe: distance <= minimum_safe_distance.
        Slowdown: minimum_safe_distance < distance <= slowdown_distance.
        Beyond slowdown_distance: no restriction from that wall.
        """
        self._validate_wall_vectors()
        unsafe = self.wall_distances <= self.minimum_safe_distance
        slowdown = (~unsafe) & (self.wall_distances <= self.slowdown_distance)
        return unsafe, slowdown

    def slow_down(self, state: DroneState, dt: float,
                  desired_velocity: ArrayLike | None = None) -> NDArray[np.float64]:
        """Ask the controller to reduce motion toward all slowdown-zone walls.

        Prepare to stop before the minimum safe distance, accounting for
        current velocity and simultaneous constraints from multiple walls.
        """
        return self._avoid(state, dt, desired_velocity, recover = False)

    def stop_and_move_away(self, state: DroneState, dt: float,
                           desired_velocity: ArrayLike | None = None) -> NDArray[np.float64]:
        """Ask the controller to stop approach and recover safe clearance.

        Resolve all nearby wall constraints together; do not correct away
        from one wall by commanding motion into another.
        """
        return self._avoid(state, dt, desired_velocity, recover = True)

    def enforce_safety(self, state: DroneState, dt: float,
                       desired_velocity: ArrayLike | None = None) -> NDArray[np.float64]:
        """Return a safe velocity and override PID commands when necessary.

        Supply navigation's desired world velocity to prevent commands toward
        walls even when currently stationary. Defaults to measured velocity.
        Braking can start outside the 0.5 m band if stopping distance requires it.
        """
        unsafe, _ = self.check_safety_zones()
        if np.any(unsafe):
            return self.stop_and_move_away(state, dt, desired_velocity)
        return self.slow_down(state, dt, desired_velocity)

    def _velocity_limits(self, desired, dt, recover):
        """Build simultaneous constraints: direction dot velocity <= limit."""
        gap = self.wall_distances - self.minimum_safe_distance
        clearance = np.maximum(gap, 0.0)
        # Stop before consuming clearance, with earlier braking at high speed.
        limits = np.minimum(np.sqrt(2 * self.braking_deceleration * clearance),
                            clearance / dt)
        approaching = np.maximum(self.wall_directions @ desired, 0.0)
        in_band = (gap > 0) & (gap <= self.SLOWDOWN_THRESHOLD)
        limits[in_band] = np.minimum(limits[in_band],
            approaching[in_band] * gap[in_band] / self.SLOWDOWN_THRESHOLD)
        if recover:
            unsafe = gap <= 0
            limits[unsafe] = -np.minimum(self.recovery_speed,
                (self.slowdown_distance - self.wall_distances[unsafe]) / self.response_time)
        return limits

    def _project_velocity(self, desired, limits):
        """Closest feasible 3D velocity, considering up to 3 active planes.

        At a Euclidean projection in 3D, at most three independent constraint
        normals are needed. Enumerating them handles corners and opposing walls
        without summing repulsion vectors that could cancel or violate a wall.
        Return None when the requested clearance recovery is infeasible.
        """
        normals = self.wall_directions
        tolerance = 1e-9
        if np.all(normals @ desired <= limits + tolerance):
            return desired.copy()
        best, best_cost = None, np.inf
        for size in range(1, min(3, len(limits)) + 1):
            for indices in combinations(range(len(limits)), size):
                indices = list(indices)
                # Orthogonalize up to three normals instead of invoking a
                # general matrix solver for this small geometric problem.
                candidate = desired.copy()
                basis = []
                for index in indices:
                    normal = normals[index]
                    perpendicular = normal.copy()
                    for axis in basis:
                        perpendicular -= np.dot(perpendicular, axis) * axis
                    length_squared = np.dot(perpendicular, perpendicular)
                    if length_squared < 1e-12:
                        break  # Dependent subsets are covered by smaller ones.
                    candidate += ((limits[index] - np.dot(normal, candidate))
                                  / length_squared) * perpendicular
                    basis.append(perpendicular / np.sqrt(length_squared))
                if len(basis) != size:
                    continue
                if np.any(normals @ candidate > limits + tolerance):
                    continue
                cost = np.sum((candidate - desired)**2)
                if cost < best_cost:
                    best, best_cost = candidate, cost
        return best

    def _avoid(self, state, dt, desired_velocity, recover):
        dt = positive(dt, "Time step")
        self._validate_wall_vectors()
        velocity = vector(state.velocity, 3, "State velocity")
        desired = velocity if desired_velocity is None else vector(
            desired_velocity, 3, "Desired velocity")
        limits = self._velocity_limits(desired, dt, recover)
        safe_velocity = self._project_velocity(desired, limits)
        self.recovery_blocked = safe_velocity is None
        if self.recovery_blocked:
            # No direction clears every wall: prevent further approach instead.
            safe_velocity = self._project_velocity(desired, np.maximum(limits, 0.0))
            if safe_velocity is None:
                safe_velocity = np.zeros(3)
        self.last_safe_velocity = safe_velocity.copy()
        # Measured approach also triggers braking when navigation requests hold.
        needs_control = (np.any(self.wall_distances <= self.slowdown_distance)
                         or np.any(self.wall_directions @ velocity > limits + 1e-9)
                         or not np.allclose(safe_velocity, desired, rtol = 0, atol = 1e-9))
        if needs_control:
            self._command_controller(state, dt, safe_velocity)
        return safe_velocity

    def _command_controller(self, state, dt, safe_velocity):
        """Generate motor commands with the same PID chain as main.control_step."""
        height, orientation, position = self.controller
        location = vector(state.position, 3, "State position")
        yaw = vector(state.orientation, 3, "State orientation")[2]
        # A zero position error removes the old waypoint's pull into a wall.
        # Reset stored PID error so it cannot continue pushing toward obstacles.
        height.integral_error = 0.0
        position.x_integral = 0.0
        position.y_integral = 0.0
        base = height.calculate_motor_speed(location[2], state, dt, safe_velocity[2])
        target = position.calculate_target_orientation(
            location[:2], yaw, state, dt, safe_velocity[:2])
        factor = max(np.cos(state.orientation[0]) * np.cos(state.orientation[1]), 0.5)
        state.motor_speeds = orientation.calculate_motor_speed(base / np.sqrt(factor), target, state, dt)
