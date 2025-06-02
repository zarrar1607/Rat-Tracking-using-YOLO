import cv2
import time
import torch
from ultralytics import YOLO
import os
import csv

program_start_time = time.time()
print(f"Program started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(program_start_time))}")

# --- CUDA Configuration ---
if torch.cuda.is_available():
    device_to_use = "cuda"
    print("CUDA is available. Using GPU for inference.")
    # Optional: print CUDA device name
    if torch.cuda.current_device() is not None: # Check if a CUDA device is actually selected
         print(f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device_to_use = "cpu"
    print("CUDA not available. Falling back to CPU for inference.")

# Load trained YOLOv8 model
model_path = "runs/detect/train30/weights/best.pt"  # Adjust model path if needed
if not os.path.exists(model_path):
    print(f"Error: Model path '{model_path}' does not exist. Please provide a valid path.")
    exit()
model = YOLO(model_path)
print(f"YOLO model '{model_path}' loaded.")

# Path to input video (the last assignment will be used)
input_video_path = "Video/3_mice.mp4"
input_video_path = "Video/brown_rats.mp4"
input_video_path = "Video/TestFile_video.mp4"
input_video_path = "Video/desktop.avi"
input_video_path = "Video/New_8229-3-4-25.mp4" # Make sure this video file exists
input_video_path = "Video/Feedback.mp4" 

if not os.path.exists(input_video_path):
    print(f"Error: Video path '{input_video_path}' does not exist. Please provide a valid path.")
    exit()

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{input_video_path}'.")
    exit()


# CSV setup
csv_path = "feedback_trajectory_full.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "x_center", "y_center"]) # Write header

    frame_idx = 0
    print(f"Starting video processing using device: {device_to_use.upper()}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert BGR frame (OpenCV default) to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference and measure time
        start_time_inference = time.time()
        results = model(rgb_frame, device=device_to_use, verbose=False)  # Use selected device, verbose=False
        end_time_inference = time.time()
        inference_time = end_time_inference - start_time_inference

        # Access detection results
        detection_result = results[0]
        boxes = detection_result.boxes  # Get bounding boxes
        
        # Filter boxes with confidence > 0.2 (adjust as needed)
        filtered_boxes = [box for box in boxes if box.conf.item() > 0.2]

        if filtered_boxes:
            # Find the box with the highest confidence for CSV and marker
            top_box = max(filtered_boxes, key=lambda b: b.conf.item())
            x1, y1, x2, y2 = top_box.xyxy[0].cpu().numpy()
            
            xm = int((x1 + x2) / 2)
            ym = int((y1 + y2) / 2)
            writer.writerow([frame_idx, xm, ym]) # Write center of top detection

            # Draw marker for the top detection
            marker_color = (0, 0, 255)  # Red color in BGR
            marker_radius = 5           # Radius of the marker circle
            marker_thickness = -1       # Filled circle
            cv2.circle(frame, (xm, ym), marker_radius, marker_color, marker_thickness)
        
        # Draw bounding boxes on the frame for ALL filtered boxes
        # This loop will not execute if filtered_boxes is empty, thus avoiding the error.
        for box_to_draw in filtered_boxes: # Iterate over all filtered boxes
            bbox = box_to_draw.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = box_to_draw.conf.item()  # Confidence score

            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw green box
            cv2.putText(frame, f"Conf: {conf:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display inference time and device on the frame
        cv2.putText(frame, f"Infer: {inference_time:.3f}s ({device_to_use.upper()})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.putText(frame, f"Frame: {frame_idx}", (10, 60), # Display current frame index
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the processed frame
        cv2.imshow("YOLO Inference", frame)
        
        frame_idx += 1 # Increment frame index for the next loop

        # Wait for 1 ms and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting display loop by user command.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
# csvfile is automatically closed due to 'with' statement

print(f"Video processing complete. Trajectory saved to '{csv_path}'. Display closed.")
program_end_time = time.time()
program_duration = program_end_time - program_start_time

print(f"Program ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(program_end_time))}")
print(f"Total program execution time: {program_duration:.2f} seconds")