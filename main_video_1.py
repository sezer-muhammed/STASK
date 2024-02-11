from utils import *
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import datetime
import pytz
import csv
import os

points_v1 = [
    [480, 345], 
    [539, 656], 
    [609, 332], 
    [721, 333], 
    [1008, 678]  
]


points_v2 = [
    [496, 293], 
    [1034, 650], 
    [638, 287], 
    [544, 696], 
    [755, 288]  
]



def get_current_tr_time(start_time: datetime.datetime, offset: datetime.timedelta) -> datetime.datetime:
    """
    Calculate the current TR time based on the offset from the start time.
    """
    return start_time + offset

def save_vehicle_pass_event(vehicle: Vehicle, timestamp: datetime.datetime):
    """
    Save the vehicle pass event to a CSV file.
    """
    filename = 'vehicle_pass_events_video_2.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['car_id', 'timestamp', 'x', 'y', 'state', 'lane'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({'car_id': vehicle.id, 'timestamp': timestamp.isoformat(), 'x': vehicle.x, 'y': vehicle.y, 
                         'state': vehicle.direction, 'lane': vehicle.lane})

def main(video_path: str):
    vehicles = {}  # Dictionary to store vehicle objects
    cap = cv2.VideoCapture(video_path)
    cross_counter = 0
    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the video

    out = cv2.VideoWriter('video_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
     
    time_increment_per_frame = datetime.timedelta(seconds=1/fps)  # Calculate the time increment per frame
    
    tracker = initialize_tracker(points_v2)
    model = YOLO('best.pt')
    
    # Set the starting TR time
    tr_timezone = pytz.timezone('Europe/Istanbul')
    start_real_time = datetime.datetime.now(tr_timezone)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame, vehicles, total_cross = process_frame(frame, tracker, model, vehicles)
        cross_counter += total_cross
        # Increment the real-world time by the time per frame
        start_real_time += time_increment_per_frame
        # Here, you would check for vehicles passing the line and log them as needed
        # Example:
        for vehicle_id, vehicle in vehicles.items():
            save_vehicle_pass_event(vehicle, start_real_time)
        cv2.imshow("Frame with Points and Lines", frame)
        out.write(frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with open('total_cross_video2.txt', 'w') as f:
        f.write(str(cross_counter))
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#save total cross value to a file
    

if __name__ == "__main__":
    video_path = 'videos/video2.MOV'
    main(video_path)