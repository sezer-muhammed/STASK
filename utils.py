import cv2
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
from optic_flow import OpticalFlowTracker
from line_finder import LineFinder
# Assuming OpticalFlowTracker, LineFinder, and YOLO classes are defined elsewhere

from dataclasses import dataclass

@dataclass
class Vehicle:
    """
    A data class for vehicle detection and tracking information.

    Attributes
    ----------
    x : float
        The current x-coordinate of the vehicle.
    y : float
        The current y-coordinate of the vehicle.
    prev_x : float
        The previous x-coordinate of the vehicle.
    prev_y : float
        The previous y-coordinate of the vehicle.
    id : int
        The unique identifier for the vehicle.
    direction : str
        The current direction of the vehicle movement.
    lane : int
        The current lane of the vehicle.
    prev_lane : int
        The previous lane of the vehicle.
    """
    x: float
    y: float
    prev_x: float = 0.0
    prev_y: float = 0.0
    id: int = 0
    direction: str = 'Unknown'
    lane: int = 0
    prev_lane: int = 0

def determine_lane_change_direction(current_lane: str, prev_lane: str) -> str:
    """
    Determines the direction of a lane change based on current and previous lanes.

    Parameters:
    - current_lane: The current lane of the vehicle.
    - prev_lane: The previous lane of the vehicle.

    Returns:
    - A string indicating the direction of the lane change ('left_lane_change', 'right_lane_change', or 'no_change').
    """
    lane_order = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3']
    
    if current_lane not in lane_order or prev_lane not in lane_order:
        return 'no_change'  # If either lane is not recognized, consider no change

    current_index = lane_order.index(current_lane)
    prev_index = lane_order.index(prev_lane)
    if current_index < prev_index:
        return 'left_lane_change' if 'L' in current_lane else 'right_lane_change'
    elif current_index > prev_index:
        return 'right_lane_change' if 'L' in current_lane else 'left_lane_change'
    else:
        return 'no_change'


def has_crossed_line(prev_point: Tuple[float, float], current_point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
    """
    Determines if a point has crossed from one side of a line to the other.

    Parameters
    ----------
    prev_point : Tuple[float, float]
        The previous coordinates of the point.
    current_point : Tuple[float, float]
        The current coordinates of the point.
    line : Tuple[Tuple[float, float], Tuple[float, float]]
        The coordinates defining the line.

    Returns
    -------
    bool
        True if the point has crossed the line, False otherwise.
    """
    # Line equation coefficients A*x + B*y + C = 0
    A = line[1][1] - line[0][1]
    B = line[0][0] - line[1][0]
    C = A * line[0][0] + B * line[0][1]
    
    prev_position = A * prev_point[0] + B * prev_point[1] - C
    current_position = A * current_point[0] + B * current_point[1] - C
    
    # Check if prev and current positions are on different sides of the line
    return prev_position * current_position < 0

def point_to_line_distance(point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
    Calculate the distance between a point and a line defined by two points.

    Parameters
    ----------
    point : Tuple[float, float]
        The coordinates of the point (x0, y0).
    line : Tuple[Tuple[float, float], Tuple[float, float]]
        The coordinates of the two points defining the line ((x1, y1), (x2, y2)).

    Returns
    -------
    float
        The distance between the point and the line.
    """
    x0, y0 = point
    (x1, y1), (x2, y2) = line
    numerator = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = numerator / denominator
    return distance

def initialize_tracker(points) -> OpticalFlowTracker:
    """
    Initializes the OpticalFlowTracker with predefined points.

    Returns
    -------
    OpticalFlowTracker
        An instance of OpticalFlowTracker initialized with points.
    """

    tracker = OpticalFlowTracker(points)
    return tracker

def process_frame(frame: np.ndarray, tracker: OpticalFlowTracker, model: YOLO, vehicles: Dict[int, Vehicle]) -> np.ndarray:
    """
    Process a single frame: tracking, line finding, and vehicle lane detection.

    Parameters
    ----------
    frame : np.ndarray
        The current video frame.
    tracker : OpticalFlowTracker
        An instance of OpticalFlowTracker.
    model : YOLO
        An instance of the YOLO model for object detection.

    Returns
    -------
    np.ndarray
        The processed frame with visual annotations.
    """
    updated_points = tracker.process_frame(frame)
    lines = []
    if len(updated_points) >= 5:
        line_finder = LineFinder(np.array(updated_points))
        lines = line_finder.find_lines()
        for i, line in enumerate(lines):
            cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 0, 0), 2)
            cv2.putText(frame, str(i), tuple(map(int, line[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for point in updated_points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    frame, vehicles, total_cross = detect_and_annotate_vehicles(frame, model, lines, vehicles)
    return frame, vehicles, total_cross

def point_relative_to_line(point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> str:
    """
    Determines if a point is to the left or right of a line.

    Parameters:
    - point: A tuple representing the coordinates of the point (x, y).
    - line: A tuple representing the line as two points ((x1, y1), (x2, y2)).

    Returns:
    - 'left' if the point is to the left of the line,
    - 'right' if the point is to the right,
    - 'on' if the point is on the line.
    """
    (x, y) = point
    ((x1, y1), (x2, y2)) = line
    determinant = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
    if determinant > 0:
        return 'left'
    elif determinant < 0:
        return 'right'
    else:
        return 'on'

def get_lane_name(vehicle_position: Tuple[float, float], lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], lane_index) -> str:
    """
    Determines the lane name based on vehicle position relative to detected lines.

    Parameters:
    - vehicle_position: The (x, y) coordinates of the vehicle.
    - lines: A list of lines, each represented as two points.
    - lane_index: The index of the lane as determined by distance.

    Returns:
    - The lane name as a string (e.g., 'L1', 'L2', 'L3', 'R1', 'R2', 'R3').
    """
    if lane_index == "m":  # Middle lane, decide based on the side relative to the first line
        side = point_relative_to_line(vehicle_position, lines[1])
        return 'L2' if side == 'left' else 'R2'
    else:
        # Assuming lines[0] is the reference line for L1/R1 and L3/R3 decisions
        side = point_relative_to_line(vehicle_position, lines[1])
        if lane_index == 1:
            return 'L1'
        elif lane_index == 2:
            return 'L3' if side == 'left' else 'R1'
        elif lane_index == 3:
            return 'R3'
        else:
            raise ValueError("Invalid lane index")

def detect_and_annotate_vehicles(frame: np.ndarray, model: YOLO, lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], vehicles: Dict[int, Vehicle]) -> Tuple[np.ndarray, Dict[int, Vehicle]]:
    """
    Detects vehicles in the frame, updates their information in a dictionary of Vehicle instances,
    and annotates their detected lanes on the frame.

    Parameters
    ----------
    frame : np.ndarray
        The current video frame.
    model : YOLO
        An instance of the YOLO model for object detection.
    lines : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        The lines detected in the current frame.
    vehicles : Dict[int, Vehicle]
        A dictionary of Vehicle instances keyed by their IDs.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, Vehicle]]
        The frame with vehicles and their lanes annotated, and the updated dictionary of Vehicle instances.
    """
    total_cross = 0
    updated_vehicles = {}  # Temporary dictionary to track updates in this frame
    results = model.track(frame, persist=True)
    check_line = lines[3]
    for box in results[0].boxes:
        if box.cls == 3:  # Assuming class '3' is for vehicles
            vehicle_id = int(box.id)  # Assuming the model provides a unique ID for each detected box
            box_center = box.xywh[0, :2].numpy()

            distances = [point_to_line_distance(box_center, line) for line in lines[:3]]
            if distances:
                distances_copy = distances.copy()
                distances_copy.sort()
                difference = distances_copy[0] / (distances_copy[1] + distances_copy[0])
                lane = "m" if difference > 0.372 else distances.index(min(distances)) + 1  # Assuming lane '2' is the middle lane
                lane_name = get_lane_name(box_center, lines, lane)
                # Check if the vehicle was previously tracked
                if vehicle_id in vehicles.keys():
                    prev_x, prev_y, prev_lane = vehicles[vehicle_id].x, vehicles[vehicle_id].y, vehicles[vehicle_id].lane
                    direction = 'UP' if prev_y > box_center[1] else 'DOWN'
                    crossed_line = has_crossed_line((prev_x, prev_y), box_center, check_line)
                    if crossed_line:
                        total_cross += 1
                else:
                    prev_x, prev_y, prev_lane, direction = 0, 0, 0, 'UNKNOWN'
                        
                updated_vehicles[vehicle_id] = Vehicle(x=box_center[0], y=box_center[1], prev_x=prev_x, prev_y=prev_y, id=vehicle_id, direction=direction, lane=lane_name, prev_lane=prev_lane)

                lane_change_direction = determine_lane_change_direction(updated_vehicles[vehicle_id].lane, updated_vehicles[vehicle_id].prev_lane)

                if lane_change_direction in ['left_lane_change', 'right_lane_change']:
                    updated_vehicles[vehicle_id].direction = lane_change_direction

                cv2.circle(frame, (int(box_center[0]), int(box_center[1])), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'{updated_vehicles[vehicle_id].lane}, {updated_vehicles[vehicle_id].direction}', (int(box_center[0]), int(box_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Remove vehicles not updated in this frame
    vehicles = updated_vehicles

    return frame, vehicles, total_cross