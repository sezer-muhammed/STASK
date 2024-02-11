#write code read all videos in videos folder and convert them to jpgs and save them in jpgs folder, convert 1 frame each 200

import cv2
import os

def video_to_jpg(video_path, jpg_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        if count%200 == 0:
            cv2.imwrite(jpg_path + "/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
    print('Total frames:', count)
    vidcap.release()

def main():
    video_folder = "videos"
    jpg_folder = "jpgs"
    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)
    for video in os.listdir(video_folder):
        video_path = video_folder + "/" + video
        jpg_path = jpg_folder + "/" + video.split(".")[0]
        if not os.path.exists(jpg_path):
            os.makedirs(jpg_path)
        video_to_jpg(video_path, jpg_path)

if __name__ == "__main__":
    main()