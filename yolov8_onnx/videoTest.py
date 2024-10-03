import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a pretrained YOLOv8s model
model = YOLO("models/best3.pt")

# Define path to video file
source = "test.mp4"

# Open the video file
cap = cv2.VideoCapture(source)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = cap.get(cv2.CAP_PROP_FPS)
fps = 25

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Run inference on the source
results = model(source, stream=True)

# Process each frame
for result in results:
    frame = result.plot()
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")