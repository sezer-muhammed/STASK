import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self, points):
        """
        Initialize the OpticalFlowTracker with points to track.

        Parameters:
        points (list): List of points [(x1, y1), (x2, y2), ...] to track.
        """
        self.points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None

    def process_frame(self, frame):
        """
        Process a single frame for optical flow tracking.

        Parameters:
        frame (np.array): The current frame to process.

        Returns:
        np.array: The updated points after tracking.
        """
        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            # Return the original points if it's the first frame
            return self.points.reshape(-1, 2)

        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.points, None, **self.lk_params)

        updated_points = np.empty((0, 2), dtype=np.float32)
        if new_points is not None and status is not None:
            # Select good points
            good_new = new_points[status == 1]

            # Update the points
            self.points = good_new.reshape(-1, 1, 2)
            updated_points = good_new.reshape(-1, 2)
        else:
            print("Optical flow couldn't be calculated for some points.")

        self.prev_gray = gray  # Update the previous frame
        return updated_points