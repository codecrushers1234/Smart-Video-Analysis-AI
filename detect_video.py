from ultralytics import YOLO
import cv2

# Load pre-trained YOLO AI model
model = YOLO("yolov8n.pt")

# Input and output video names
input_video = "input_video.mp4"
output_video = "output_video.mp4"

# Read input video
cap = cv2.VideoCapture(input_video)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # AI detects objects
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Save frame to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print("Video analysis completed successfully")
