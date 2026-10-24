"""Three A* outlet missions followed by a return flight and landing at home."""

import numpy as np

from drone_sim.configuration import parameters
from drone_sim.validation import positive


def run_outlet_inspections(duration_per_mission = 120.0, verbose = True, terrain = None):
    # Import lazily so the normal single-mission entry points remain reusable.
    from drone_sim import main

    duration_per_mission = positive(duration_per_mission, "Mission time limit")
    room = main.create_terrain() if terrain is None else terrain
    if len(room.outlets) != 3:
        raise ValueError("The inspection tour requires exactly three wall outlets.")
    inspection_range = positive(parameters.OUTLET_INSPECTION_RANGE, "Inspection range")
    hold_time = positive(parameters.OUTLET_INSPECTION_HOLD_TIME, "Inspection hold time")
    points = [outlet.inspection_point(room, parameters.OUTLET_INSPECTION_STANDOFF,
              room.base_height + parameters.OUTLET_INSPECTION_HEIGHT, inspection_range)
              for outlet in room.outlets]
    drone = main.create_drone(room)
    home_position = drone.initial_position.copy()
    state, histories, inspections = None, [], []
    return_mission = None
    time_offset = 0.0
    try:
        for index, (outlet, point) in enumerate(zip(room.outlets, points), start = 1):
            yaw = outlet.facing_yaw()

            def ready(current_state):
                distance = np.linalg.norm(current_state.position - outlet.position)
                yaw_error = np.arctan2(np.sin(current_state.orientation[2] - yaw),
                                       np.cos(current_state.orientation[2] - yaw))
                return distance <= inspection_range and abs(yaw_error) <= np.radians(5)

            # Each call creates its own A* search, route and path follower.
            # Later legs start at the measured state without resetting physics.
            state, history = main._run_mission(
                drone, None, duration_per_mission, False, False, yaw, room,
                point, state, None, None, initialize_drone = index == 1,
                arrival_check = ready, arrival_wait_time = hold_time,
                route_planning_mode = "astar")
            completed = history["stop_reason"] == "destination_reached" and ready(state)
            distance = float(np.linalg.norm(state.position - outlet.position))
            inspections.append({
                "mission_id": index, "outlet": outlet.name,
                "outlet_position": np.asarray(outlet.position),
                "inspection_point": point.copy(), "facing_yaw": yaw,
                "actual_position": state.position.copy(), "distance": distance,
                "completed": completed, "start_time": time_offset,
                "inspection_started_at": (None if history["arrival_time"] is None
                    else time_offset + history["arrival_time"]),
                "end_time": time_offset + history["time"][-1],
                "planned_waypoints": history["waypoints"].copy(),
            })
            histories.append(history)
            time_offset += history["time"][-1]
            if verbose:
                print(f"A* mission {index}: {outlet.name}, distance {distance:.3f} m, "
                      f"inspection {'complete' if completed else 'incomplete'}")
            if not completed:
                break  # Never claim or attempt later inspections after failure.
        if len(inspections) == 3 and all(item["completed"] for item in inspections):
            state, history = main._run_mission(
                drone, None, duration_per_mission, True, False, 0.0, room,
                home_position, state, None, None, initialize_drone = False,
                route_planning_mode = "astar", landing_mode = "exact")
            completed = (history["stop_reason"] == "destination_reached"
                         and history["landed"]
                         and np.linalg.norm(state.position[:2] - home_position[:2])
                         <= parameters.ARRIVAL_TOLERANCE)
            return_mission = {
                "completed": bool(completed), "home_position": home_position.copy(),
                "actual_position": state.position.copy(), "start_time": time_offset,
                "end_time": time_offset + history["time"][-1],
                "planned_waypoints": history["waypoints"].copy(),
            }
            histories.append(history)
            if verbose:
                print(f"Return home: {'landed at start' if completed else 'incomplete'}")
    finally:
        drone.close()

    data = _combine_histories(histories, inspections, return_mission, home_position)
    if verbose:
        print(f"Outlets inspected: {sum(item['completed'] for item in inspections)}/3")
    return state, data


def _combine_histories(histories, inspections, return_mission, home_position):
    """Keep one timeline for graphs while retaining the separate mission data."""
    data = histories[-1].copy()
    sample_keys = [key for key, value in histories[0].items()
                   if isinstance(value, list) and len(value) == len(histories[0]["time"])]
    for key in sample_keys:
        data[key] = [value for history in histories for value in history[key]]
    data["clearance"] = np.concatenate([history["clearance"] for history in histories])
    data["time"], data["waypoint_index"] = [], []
    time_offset, waypoint_offset = 0.0, 0
    data["stage_start_times"] = {}
    for index, history in enumerate(histories):
        data["time"].extend(time_offset + np.asarray(history["time"]))
        data["waypoint_index"].extend(waypoint_offset + np.asarray(history["waypoint_index"]))
        if index < len(inspections):
            inspection = inspections[index]
            mission_id = inspection["mission_id"]
            data["stage_start_times"][f"mission_{mission_id}"] = time_offset
            if inspection["inspection_started_at"] is not None:
                data["stage_start_times"][f"inspect_{mission_id}"] = inspection["inspection_started_at"]
        else:
            data["stage_start_times"]["return_home"] = time_offset
            for stage in ("landing", "landed"):
                if stage in history["stage_start_times"]:
                    data["stage_start_times"][stage] = time_offset + history["stage_start_times"][stage]
        time_offset += history["time"][-1]
        waypoint_offset += len(history["waypoints"])
    data["waypoints"] = np.concatenate([history["waypoints"] for history in histories])
    data["inspections_complete"] = len(inspections) == 3 and all(item["completed"] for item in inspections)
    data["return_complete"] = return_mission is not None and return_mission["completed"]
    data["mission_complete"] = data["inspections_complete"] and data["return_complete"]
    data["stop_reason"] = "inspection_and_return_complete" if data["mission_complete"] else histories[-1]["stop_reason"]
    data["arrival_time"] = (return_mission["start_time"] + histories[-1]["arrival_time"]
                            if return_mission is not None and histories[-1]["arrival_time"] is not None
                            else None)
    data["return_mission"] = return_mission
    data["home_position"] = home_position.copy()
    data["outlet_inspections"] = inspections
    data["mission_histories"] = histories
    data["inspection_range"] = parameters.OUTLET_INSPECTION_RANGE
    data["inspection_hold_time"] = parameters.OUTLET_INSPECTION_HOLD_TIME
    return data
