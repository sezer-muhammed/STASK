import numpy as np
import cv2

class LineFinder:
    def __init__(self, points):
        """
        Initializes the LineFinder with a set of updated points.

        Parameters:
        points (np.ndarray): Array of updated points [(x1, y1), (x2, y2), ...].
        """
        if points.shape[0] == 5:
            # Add a point at (0, 0) if there are only 5 points
            self.points = np.vstack([[[0, 720]], points])
        else:
            self.points = points

    def sort_points(self):
        """
        Sorts points for line calculations based on their x and y coordinates.
        """
        # Sort by y to separate top and bottom points
        top_points = self.points[np.argsort(self.points[:, 1])[:3]]
        bottom_points = self.points[np.argsort(self.points[:, 1])[-3:]]

        # Sort by x to identify leftmost and rightmost points
        leftmost_top = top_points[np.argmin(top_points[:, 0])]
        rightmost_top = top_points[np.argmax(top_points[:, 0])]
        leftmost_bottom = bottom_points[np.argmin(bottom_points[:, 0])]
        rightmost_bottom = bottom_points[np.argmax(bottom_points[:, 0])]

        # Identify middle points
        middle_top = top_points[np.argsort(top_points[:, 0])[1]]
        middle_bottom = bottom_points[np.argsort(bottom_points[:, 0])[1]]

        return leftmost_top, leftmost_bottom, middle_top, middle_bottom, rightmost_top, rightmost_bottom

    def find_lines(self):
        """
        Finds and returns four lines based on the updated points.

        Returns:
        list of tuples: Each tuple represents a line defined by two points (start, end).
        """
        leftmost_top, leftmost_bottom, middle_top, middle_bottom, rightmost_top, rightmost_bottom = self.sort_points()

        # Calculate the average y coordinate for the horizontal line
        avg_y = np.mean([leftmost_top[1], leftmost_bottom[1], middle_top[1], middle_bottom[1], rightmost_top[1], rightmost_bottom[1]])

        # Define lines
        line1 = (leftmost_top, leftmost_bottom)
        line2 = (middle_top, middle_bottom)
        line3 = (rightmost_top, rightmost_bottom)
        line4 = ((np.min(self.points[:, 0]), avg_y), (np.max(self.points[:, 0]), avg_y))  # Horizontal line

        return [line1, line2, line3, line4]
