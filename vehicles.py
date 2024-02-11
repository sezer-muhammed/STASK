from dataclasses import dataclass

@dataclass
class DataPoint:
    """
    A data class representing a point with additional attributes.

    Attributes:
        x (str): The x-coordinate of the point.
        y (str): The y-coordinate of the point.
        id (str): The id of the point.
        prev_x (str): The previous x-coordinate of the point.
        prev_y (str): The previous y-coordinate of the point.
        direction (str): The direction of the point.
        lane (str): The lane of the point.
    """
    x: int
    y: int
    id: int
    prev_x: int
    prev_y: int
    direction: str
    lane: str