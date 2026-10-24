import numpy as np
from drone_sim.configuration.project_paths import output_path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch, Rectangle
from drone_sim.physics.collision_check import CollisionCheck
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from drone_sim.world.terrain import RoomTerrain

# Terrain preview and wider scale for the 3D graph (metres).
TERRAIN_PLOT_SPAN = 16.0


def use_backend(name):
    """Select a rendering backend, for example Agg for unattended tuning."""

    plt.switch_backend(name)


def close_all():

    plt.close("all")


def landing_label(simulation_data):
    """Describe the landing mode and any selected nearby destination."""

    if "outlet_inspections" in simulation_data:
        completed = sum(item["completed"] for item in simulation_data["outlet_inspections"])
        home_status = "landed at start" if simulation_data.get("return_complete") else "not complete"
        return (f"Physics backend: {simulation_data['physics_backend']}\n"
                f"Outlet inspections: {completed}/3 — return home: {home_status}\n"
                f"Maximum inspection range: {simulation_data['inspection_range']:g} m")

    mode = simulation_data.get("landing_mode", "unspecified")
    label = f"Landing mode: {mode if mode is not None else 'disabled'}"
    if "physics_backend" in simulation_data:
        label = f"Physics backend: {simulation_data['physics_backend']}\n" + label

    if mode == "safe_nearby" and "selected_destination" in simulation_data:
        x, y = simulation_data["selected_destination"][:2]
        label += f"\nSelected landing XY: ({x:.2f}, {y:.2f}) m"

    return label


def mark_flight_stages(axes, stage_start_times):
    """Mark actual mission stage starts consistently across time-series panels."""

    colors = {"climbing": "#567a35", "flying": "#206593",
              "landing": "#aa6632", "landed": "#656565"}
    colors.update({f"mission_{i}": colour for i, colour in
                   enumerate(("tab:orange", "tab:green", "tab:purple"), start = 1)})
    colors.update({f"inspect_{i}": "#656565" for i in range(1, 4)})
    colors["return_home"] = "tab:brown"
    events = [(stage, time) for stage, time in stage_start_times.items() if stage in colors]

    for ax in axes:
        right_edge = ax.get_xlim()[1]

        for index, (stage, start_time) in enumerate(events):
            if stage not in colors:
                continue

            color = colors[stage]
            ax.axvline(start_time, color = color, linestyle = ":", linewidth = 1.2, alpha = 0.8, gid = f"stage-{stage}")
            end_time = events[index + 1][1] if index + 1 < len(events) else right_edge
            label_time = (start_time + end_time) / 2
            ax.annotate(stage.capitalize(), (label_time, 1.0),
                        xycoords = ax.get_xaxis_transform(), xytext = (0, 4),
                        textcoords = "offset points", color = color, fontsize = 8,
                        ha = "center", va = "bottom", annotation_clip = False)

        ax.set_title(ax.get_title(), pad = 26)


def plot_data(simulation_data, *, show = True):

    time_history = np.asarray(simulation_data["time"], dtype = float)
    height_history = np.asarray(simulation_data["height"], dtype = float)
    velocity_history = np.asarray(simulation_data["velocity"], dtype = float)
    position_history = np.asarray(simulation_data["position"], dtype = float)
    motor_speed_history = np.asarray(simulation_data["motor_speed"], dtype = float)
    orientation_history = np.asarray(simulation_data["orientation"], dtype = float)
    target_orientation_history = np.asarray(simulation_data["target_orientation"], dtype = float)
    target_position_history = np.asarray(simulation_data.get("commanded_position_history", simulation_data["target_position_history"]), dtype = float)

    hover_speed = simulation_data["hover_speed"]

    if target_orientation_history.ndim == 1:
        target_orientation_history = np.tile(target_orientation_history, (len(time_history), 1))

    if orientation_history.ndim != 2:
        raise ValueError("orientation_history must contain one [roll, pitch, yaw] array for every timestep.")

    figure = plt.figure(figsize = (12, 12))
    figure.suptitle(landing_label(simulation_data), fontsize = 13)

    # Plot 1: Altitude
    plt.subplot(3, 2, 1)
    plt.plot(time_history, height_history, label = "Actual height")
    plt.step(time_history, target_position_history[:, 2], color = "red", linestyle = "--", where = "post", label = "Target height")

    if "ground_height_history" in simulation_data:
        plt.plot(time_history, simulation_data["ground_height_history"], color = "olivedrab", label = "Ground elevation")

    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Drone Altitude")
    plt.grid(True)
    plt.legend()

    # Plot 2: Vertical velocity
    plt.subplot(3, 2, 2)
    plt.plot(time_history, velocity_history, color = "orange")
    plt.axhline(y = 0, color = "black", linestyle = "--")
    plt.xlabel("Time (s)")
    plt.ylabel("Vertical velocity (m/s)")
    plt.title("Vertical Velocity")
    plt.grid(True)

    # Plot 3: Four motor speeds
    plt.subplot(3, 2, 3)

    for motor_index in range(4):
        plt.plot(time_history, motor_speed_history[:, motor_index], label = f"Motor {motor_index + 1}")

    plt.axhline(y = hover_speed, color = "red", linestyle = "--", label = "Hover speed")
    plt.axhline(y = simulation_data["max_motor_speed"], color = "purple", linestyle = "--", linewidth = 1.2, label = "Maximum motor speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor speed (rad/s)")
    plt.title("Motor Speeds")
    plt.grid(True)
    plt.legend(fontsize = 8)

    # Plot 4: Roll, pitch and yaw
    plt.subplot(3, 2, 4)
    axis_names = ["Roll", "Pitch", "Yaw"]
    axis_colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index in range(3):
        plt.plot(time_history, orientation_history[:, axis_index], color = axis_colors[axis_index], label = f"Actual {axis_names[axis_index]}")
        plt.plot(time_history, target_orientation_history[:, axis_index], color = axis_colors[axis_index], linestyle = "--", label = f"Target {axis_names[axis_index]}")

    plt.axhline(y = 0, color = "black", linewidth = 0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (degrees)")
    plt.title("Orientation Tracking")
    plt.grid(True)
    plt.legend(fontsize = 7)

    # Plot 5: Horizontal position
    plt.subplot(3, 2, 5)
    plt.plot(time_history, position_history[:, 0], color = "tab:blue", label = "X position")
    plt.plot(time_history, position_history[:, 1], color = "tab:orange", label = "Y position")
    plt.step(time_history, target_position_history[:, 0], color = "tab:blue", linestyle = "--", where = "post", label = "Target X")
    plt.step(time_history, target_position_history[:, 1], color = "tab:orange", linestyle = "--", where = "post", label = "Target Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal position (m)")
    plt.title("Horizontal Position and Drift")
    plt.grid(True)
    plt.legend(fontsize = 8)

    # Plot 6: Distance to the active waypoint
    plt.subplot(3, 2, 6)
    plt.plot(time_history, simulation_data["distance_to_target"], label = "Distance to active waypoint")
    plt.axhline(y = simulation_data["arrival_tolerance"], color = "red", linestyle = "--", label = "Arrival tolerance")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Waypoint Tracking")
    plt.grid(True)
    plt.legend(fontsize = 8)

    mark_flight_stages(figure.axes, simulation_data.get("stage_start_times", {}))
    figure.subplots_adjust(hspace = 0.55)
    plt.tight_layout(h_pad = 2.0, rect = (0, 0, 1, 0.94))
    plt.savefig(output_path("altitude_control_results.png"), dpi = 300)

    if show:
        plt.show()

    return figure

def _room_safety_rectangles(terrain, simulation_data):
    """Non-overlapping red and yellow rectangular strips inside room walls."""
    minimum = float(simulation_data.get("minimum_safe_distance", terrain.minimum_safe_distance))
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("Minimum wall distance must be finite and positive.")
    xmin, xmax, ymin, ymax = terrain.plot_bounds

    def ring(inset, thickness, bounds = None):
        left, right, bottom, top = terrain.plot_bounds if bounds is None else bounds
        x0, x1, y0, y1 = left + inset, right - inset, bottom + inset, top - inset
        width, length = x1 - x0, y1 - y0
        if width <= 0 or length <= 0:
            return []
        depth = min(thickness, width / 2, length / 2)
        rectangles = [(x0, y0, width, depth), (x0, y1 - depth, width, depth)]
        if length > 2 * depth:
            rectangles.extend([(x0, y0 + depth, depth, length - 2 * depth),
                               (x1 - depth, y0 + depth, depth, length - 2 * depth)])
        return rectangles

    red, yellow = ring(0, minimum), ring(minimum, CollisionCheck.SLOWDOWN_THRESHOLD)
    for centre, half_extents in terrain.pillar_boxes():
        x, y, _ = centre
        hx, hy, _ = half_extents
        red.extend(ring(0, minimum, (x - hx - minimum, x + hx + minimum,
                                    y - hy - minimum, y + hy + minimum)))
        outer = minimum + CollisionCheck.SLOWDOWN_THRESHOLD
        yellow.extend(ring(0, CollisionCheck.SLOWDOWN_THRESHOLD,
                           (x - hx - outer, x + hx + outer,
                            y - hy - outer, y + hy + outer)))
    return [("red", red, f"Unsafe: within {minimum:g} m of walls/objects"),
            ("yellow", yellow,
             f"Slow down: next {CollisionCheck.SLOWDOWN_THRESHOLD:g} m")]


def _plot_outlet_missions(ax, simulation_data, is_3d = False):
    """Show wall outlets, their viewpoints and each independent planned leg."""
    terrain = simulation_data["terrain"]
    for index, outlet in enumerate(getattr(terrain, "outlets", ())):
        point = np.asarray(outlet.position)
        coordinates = point if is_3d else point[:2]
        ax.scatter(*coordinates, color = "deepskyblue", edgecolors = "black", marker = "s",
                   s = 65, zorder = 6, clip_on = False,
                   label = "Wall power outlet" if index == 0 else None)
        normal = np.asarray(outlet.inward_normal)
        if is_3d:
            label_point = point + normal * 0.3 + [0, 0, 0.15]
            ax.text(*label_point, f"O{index + 1}", fontsize = 8, zorder = 6)
        else:
            ax.annotate(f"O{index + 1}", coordinates,
                        xytext = (normal[0] * 24, normal[1] * 24 + (8 if normal[1] == 0 else 0)),
                        textcoords = "offset points", ha = "center", fontsize = 8)

    missions = simulation_data.get("outlet_inspections", [])
    for index, mission in enumerate(missions):
        colour = ("tab:orange", "tab:green", "tab:purple")[index % 3]
        path = np.asarray(mission["planned_waypoints"])
        coordinates = path.T if is_3d else path[:, :2].T
        ax.plot(*coordinates, color = colour, linestyle = "--", linewidth = 1.5,
                label = f"A* mission {mission['mission_id']}")
        point = np.asarray(mission["inspection_point"])
        coordinates = point if is_3d else point[:2]
        ax.scatter(*coordinates, color = colour, marker = "D", s = 40, zorder = 5,
                   label = "Inspection viewpoint" if index == 0 else None)
        connector = np.vstack((point, mission["outlet_position"]))
        ax.plot(*(connector.T if is_3d else connector[:, :2].T),
                color = colour, linestyle = ":", linewidth = 1, alpha = 0.8)
    return_mission = simulation_data.get("return_mission")
    if return_mission is not None:
        path = np.asarray(return_mission["planned_waypoints"])
        ax.plot(*(path.T if is_3d else path[:, :2].T), color = "tab:brown",
                linestyle = "--", linewidth = 1.5, label = "A* return home")
    if "home_position" in simulation_data:
        home = np.asarray(simulation_data["home_position"])
        ax.scatter(*(home if is_3d else home[:2]), color = "black", marker = "P",
                   s = 75, zorder = 6, label = "Takeoff / return point")
    return bool(missions)


def plot_3d_trajectory(simulation_data, *, show = True):

    ground_height = simulation_data["ground_height"]
    terrain = simulation_data["terrain"]
    is_room = isinstance(terrain, RoomTerrain)
    positions = np.asarray(simulation_data["position"], dtype = float)
    waypoints = np.asarray(simulation_data["waypoints"], dtype = float)

    fig = plt.figure(figsize = (10, 8))
    fig.suptitle(landing_label(simulation_data), fontsize = 13)
    ax = fig.add_subplot(111, projection = "3d", computed_zorder = False)

    all_points = np.vstack((positions, waypoints))
    requested_route = simulation_data.get("requested_waypoints") if simulation_data.get("landing_mode") == "safe_nearby" else None

    if requested_route is not None:
        requested_route = np.asarray(requested_route, dtype = float)
        all_points = np.vstack((all_points, requested_route))
    map_points = np.vstack((all_points[:, :2], getattr(terrain, "center", all_points[0, :2])))

    if hasattr(terrain, "plot_bounds"):
        xmin, xmax, ymin, ymax = terrain.plot_bounds
        map_points = np.vstack((map_points, [[xmin, ymin], [xmax, ymax]]))

    lower = map_points.min(axis = 0)
    upper = map_points.max(axis = 0)
    center = (lower + upper) / 2.0
    span = max(TERRAIN_PLOT_SPAN, float(np.max(upper - lower)) * (1.0 if hasattr(terrain, "plot_bounds") else 1.3))
    if is_room:
        span = max(terrain.width, terrain.length) * 1.05
    x_limits = (center[0] - span / 2, center[0] + span / 2)
    y_limits = (center[1] - span / 2, center[1] + span / 2)
    terrain_x, terrain_y = np.meshgrid( np.linspace(*x_limits, 220), np.linspace(*y_limits, 220))
    terrain_z = terrain.height(terrain_x, terrain_y)
    terrain_colors = LinearSegmentedColormap.from_list( "landscape", ["#315646", "#6b8c58", "#aaa178", "#a9957b", "#c4b497"])
    colors = LightSource(azdeg = 315, altdeg = 45).shade(
        terrain_z, cmap = terrain_colors, vert_exag = 1.0, blend_mode = "soft",
        dx = span / 219, dy = span / 219,
        vmin = float(np.min(terrain_z)),
        vmax = max(float(np.max(terrain_z)), float(np.min(terrain_z)) + 0.01))
    if is_room:
        colors[..., :3] = 0.94
    ax.plot_surface(terrain_x, terrain_y, terrain_z, facecolors = colors, linewidth = 0, antialiased = False, shade = False, rcount = 220, ccount = 220, zorder = 1)

    if is_room:
        xmin, xmax, ymin, ymax = terrain.plot_bounds
        floor, top = terrain.base_height, terrain.base_height + terrain.wall_height
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        faces = [[(a[0], a[1], floor), (b[0], b[1], floor),
                  (b[0], b[1], top), (a[0], a[1], top)]
                 for a, b in zip(corners, corners[1:] + corners[:1])]
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors = "lightsteelblue", edgecolors = "slategrey", alpha = 0.2))
        for centre, half_extents in terrain.pillar_boxes():
            x, y, z = centre
            hx, hy, hz = half_extents
            bottom = [(x - hx, y - hy, z - hz), (x + hx, y - hy, z - hz),
                      (x + hx, y + hy, z - hz), (x - hx, y + hy, z - hz)]
            top_face = [(vx, vy, z + hz) for vx, vy, _ in bottom]
            pillar_faces = [top_face] + [[bottom[i], bottom[(i + 1) % 4],
                top_face[(i + 1) % 4], top_face[i]] for i in range(4)]
            ax.add_collection3d(Poly3DCollection(
                pillar_faces, facecolors = "dimgray", edgecolors = "black", zorder = 2.5))
        for colour, rectangles, _ in _room_safety_rectangles(terrain, simulation_data):
            z = floor + 0.005
            zones = [[(x, y, z), (x + w, y, z), (x + w, y + h, z), (x, y + h, z)]
                     for x, y, w, h in rectangles]
            if zones:
                ax.add_collection3d(Poly3DCollection(
                    zones, facecolors = colour, edgecolors = "none", alpha = 0.2, zorder = 2))

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color = "tab:blue", label = "Actual trajectory")
    adapted = simulation_data.get("landing_mode") == "safe_nearby"
    if not _plot_outlet_missions(ax, simulation_data, is_3d = True):
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color = "tab:orange", linestyle = "--", marker = "o", label = "Adapted planned route" if adapted else "Planned route")

    if requested_route is not None:
        ax.plot(requested_route[:, 0], requested_route[:, 1], requested_route[:, 2], color = "tab:purple", linestyle = ":", linewidth = 2, label = "Requested route (comparison only)")

    if adapted and "requested_destination" in simulation_data:
        x, y = simulation_data["requested_destination"][:2]
        ax.scatter(x, y, float(terrain.height(x, y)), color = "tab:purple", marker = "D", s = 60, zorder = 5, label = "Requested landing site")
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color = "green", s = 80, zorder = 4, label = "First recorded position")
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], color = "red", marker = "*", s = 150, zorder = 4, label = "Selected landing site" if adapted else "Endpoint")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color = "purple", marker = "x", s = 80, zorder = 5, label = "Final position")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Drone Trajectory in Room" if is_room else "Drone Trajectory over Terrain")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    z_lower = min(float(np.min(terrain_z)), float(all_points[:, 2].min()))
    z_upper = max(ground_height + 6.0, float(all_points[:, 2].max()) + 1.0, float(np.max(terrain_z)) + 1.0)
    if is_room:
        z_upper = max(z_upper, terrain.base_height + terrain.wall_height + 0.5)
    ax.set_zlim(z_lower, z_upper)
    # Match the axes' physical spans so terrain and flight retain their proportions.
    ax.set_box_aspect((span, span, z_upper - z_lower))
    ax.view_init(elev = 28, azim = -60)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(facecolor = "lightsteelblue" if is_room else "olivedrab", alpha = 0.85)], labels + ["Room (4 m walls)" if is_room else "Terrain"], loc = "upper left", fontsize = 8)
    if is_room:
        zone_handles = [Patch(facecolor = colour, alpha = 0.2, label = label)
                        for colour, _, label in _room_safety_rectangles(terrain, simulation_data)]
        zone_handles.append(Patch(facecolor = "dimgray", label = "Pillar"))
        ax.legend(handles = handles + [Patch(facecolor = "lightsteelblue", alpha = 0.2,
                                   label = "Room (4 m walls)")] + zone_handles,
                  loc = "upper left", fontsize = 8)
    plt.tight_layout(rect = (0, 0, 1, 0.92))
    plt.savefig(output_path("drone_3d_trajectory.png"), dpi = 300)

    if show:
        plt.show()

    return fig

def plot_2d_trajectory(simulation_data, *, show = True):
    """Compare requested and adapted routes in the XY plane."""

    positions = np.asarray(simulation_data["position"], dtype = float)
    waypoints = np.asarray(simulation_data["waypoints"], dtype = float)
    adapted = simulation_data.get("landing_mode") == "safe_nearby"
    figure, ax = plt.subplots(figsize = (9, 8))
    figure.suptitle(landing_label(simulation_data), fontsize = 13)
    ax.plot(positions[:, 0], positions[:, 1], color = "tab:blue", label = "Actual trajectory")
    if not _plot_outlet_missions(ax, simulation_data):
        ax.plot(waypoints[:, 0], waypoints[:, 1], color = "tab:orange", linestyle = "--", label = "Adapted planned route" if adapted else "Planned route")
    requested_route = simulation_data.get("requested_waypoints") if adapted else None

    if requested_route is not None:
        requested_route = np.asarray(requested_route, dtype = float)
        ax.plot(requested_route[:, 0], requested_route[:, 1], color = "tab:purple", linestyle = ":", linewidth = 2, label = "Requested route (comparison only)")

    if adapted and "requested_destination" in simulation_data:
        x, y = simulation_data["requested_destination"][:2]
        ax.scatter(x, y, color = "tab:purple", marker = "D", s = 60, label = "Requested landing site")

    if adapted and simulation_data.get("requested_route_error"):
        ax.text(0.02, 0.02, "No route found to requested site", transform = ax.transAxes)

    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], color = "red", marker = "*", s = 150, label = "Selected landing site" if adapted else "Endpoint")
    ax.scatter(positions[-1, 0], positions[-1, 1], color = "black", marker = "x", s = 60, label = "Final position")
    ax.set(xlabel = "X position (m)", ylabel = "Y position (m)", title = "Drone Trajectory: Top View")
    ax.set_aspect("equal", adjustable = "datalim")
    terrain = simulation_data["terrain"]

    if hasattr(terrain, "plot_bounds"):
        xmin, xmax, ymin, ymax = terrain.plot_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable = "box")
        if isinstance(terrain, RoomTerrain):
            for centre, half_extents in terrain.pillar_boxes():
                x, y, _ = centre
                hx, hy, _ = half_extents
                ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy,
                    facecolor = "dimgray", edgecolor = "black", label = "Pillar", zorder = 1))
            for colour, rectangles, label in _room_safety_rectangles(terrain, simulation_data):
                for index, (x, y, width, height) in enumerate(rectangles):
                    ax.add_patch(Rectangle(
                        (x, y), width, height, facecolor = colour, edgecolor = "none",
                        alpha = 0.2, zorder = 0.5, label = label if index == 0 else None))
            ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                    color = "slategrey", linewidth = 3, label = "Room walls")

    ax.grid(True)
    ax.legend(fontsize = 8)
    figure.tight_layout(rect = (0, 0, 1, 0.92))
    figure.savefig(output_path("drone_2d_trajectory.png"), dpi = 300)

    if show:
        plt.show()

    return figure

def plot_validation(history, *, target_height, hover_speed, max_motor_speed, show = True):

    keys = ("time", "position", "velocity", "motor_speeds",
            "orientation", "target_orientation", "target_position")
    data = {key: np.asarray(history[key], dtype = float) for key in keys}

    time = data["time"]
    position = data["position"]
    velocity = data["velocity"]
    motor_speeds = data["motor_speeds"]

    orientation_degrees = np.degrees( data["orientation"] )

    target_orientation_degrees = np.degrees( data["target_orientation"] )

    target_position = data["target_position"]

    figure, axes = plt.subplots( 3, 2, figsize = (12, 12), )

    axes[0, 0].plot( time, position[:, 2], label = "Actual height", )
    axes[0, 0].axhline( target_height, color = "red", linestyle = "--", label = "Target height", )
    axes[0, 0].set_title("Drone Altitude")
    axes[0, 0].set_ylabel("Height (m)")
    axes[0, 0].legend()

    axes[0, 1].plot( time, velocity[:, 2], color = "tab:orange", )
    axes[0, 1].axhline( 0.0, color = "black", linestyle = "--", )
    axes[0, 1].set_title("Vertical Velocity")
    axes[0, 1].set_ylabel("Vertical velocity (m/s)")

    for motor_index in range(4):
        axes[1, 0].plot( time, motor_speeds[:, motor_index], label = f"Motor {motor_index + 1}", )

    axes[1, 0].axhline( hover_speed, color = "red", linestyle = "--", label = "Hover speed", )
    axes[1, 0].axhline( max_motor_speed, color = "purple", linestyle = "--", label = "Maximum motor speed", )
    axes[1, 0].set_title("Motor Speeds")
    axes[1, 0].set_ylabel("Motor speed (rad/s)")
    axes[1, 0].legend(fontsize = 8)

    labels = ["Roll", "Pitch", "Yaw"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index, label in enumerate(labels):
        axes[1, 1].plot( time, orientation_degrees[:, axis_index], color = colors[axis_index], label = f"Actual {label}", )
        axes[1, 1].plot( time, target_orientation_degrees[:, axis_index], color = colors[axis_index], linestyle = "--", label = f"Target {label}", )

    axes[1, 1].set_title("Orientation Tracking")
    axes[1, 1].set_ylabel("Orientation (degrees)")
    axes[1, 1].legend(fontsize = 7)

    axes[2, 0].plot( time, position[:, 0], color = "tab:blue", label = "X position", )
    axes[2, 0].plot( time, position[:, 1], color = "tab:orange", label = "Y position", )
    axes[2, 0].plot( time, target_position[:, 0], color = "tab:blue", linestyle = "--", label = "Target X", )
    axes[2, 0].plot( time, target_position[:, 1], color = "tab:orange", linestyle = "--", label = "Target Y", )
    axes[2, 0].set_title("Horizontal Position")
    axes[2, 0].set_ylabel("Position (m)")
    axes[2, 0].legend(fontsize = 8)

    axes[2, 1].axis("off")

    for axis in axes.flat:
        if axis.axison:
            axis.set_xlabel("Time (s)")
            axis.grid(True)

    figure.tight_layout()
    figure.savefig( output_path("auto_tune_results.png"), dpi = 200, )

    if show:
        plt.show()

    return figure
