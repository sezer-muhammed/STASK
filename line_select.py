import cv2
import numpy as np

# Function to handle mouse clicks
def click_and_draw(event, x, y, flags, param):
    global current_line, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_line = [(x, y)]  # Starting point of line

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_line.append((x, y))  # Ending point of line
        lines.append((np.array(current_line[0]), np.array(current_line[1])))  # Add the complete line
        cv2.line(img, current_line[0], current_line[1], (255, 0, 0), 2)  # Draw the line on the image

# Initialize global variables
lines = []  # To store lines' start and end points
current_line = []  # To store the current line being drawn
drawing = False  # True if mouse is pressed

# Open the video file
video_path = 'videos/video2.MOV'  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, img = cap.read()
img = cv2.resize(img, (1280, 720))
cap.release()  # Release the video capture object

# Setup the window and mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_and_draw)

# Display the image and wait for line selection
while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:  # Break loop when 'ESC' is pressed
        break

cv2.destroyAllWindows()

# Print out the selected lines
for idx, line in enumerate(lines):
    print(f"Line {idx+1}: Start [{line[0]}, {line[1]}]")
