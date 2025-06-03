import cv2
import time
import torch
from ultralytics import YOLO
import os
import csv
from datetime import datetime, timedelta

# =====================================
# 1) CONFIGURE START DATETIME & PATHS
# =====================================
start_datetime_str = "2025-03-06 06:00:00"  # Adjust to match frame 0
start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")

model_path       = "runs/detect/train30/weights/best.pt"
input_video_path = "Video/Feedback.mp4"
csv_path         = "feedback_trajectory_full_with_speed - Full File - Rat 2- 4-25-25 Start_video.csv"

# =====================================
# 2) PRINT PROGRAM START TIME
# =====================================
program_start_time = time.time()
print(f"Program started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(program_start_time))}")

# =====================================
# 3) CUDA / DEVICE SETUP
# =====================================
if torch.cuda.is_available():
    device_to_use = "cuda"
    print("CUDA is available. Using GPU for inference.")
    print(f"CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device_to_use = "cpu"
    print("CUDA not available. Falling back to CPU for inference.")

# =====================================
# 4) LOAD YOLO MODEL
# =====================================
if not os.path.exists(model_path):
    print(f"Error: Model path '{model_path}' does not exist.")
    exit()
model = YOLO(model_path)
model.model.eval()
print(f"Loaded YOLO model from '{model_path}'")

# =====================================
# 5) OPEN VIDEO & RETRIEVE FPS
# =====================================
if not os.path.exists(input_video_path):
    print(f"Error: Video path '{input_video_path}' does not exist.")
    exit()

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{input_video_path}'.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Video opened. FPS = {fps:.2f}\n")

# =====================================
# 6) PREPARE CSV FOR WRITING
# =====================================
# Columns: frame, x_min, y_min, x_max, y_max, x_center, y_center, speed_px_per_sec, timestamp
f = open(csv_path, "w", newline="")
writer = csv.writer(f)
writer.writerow([
    "frame",
    "x_min", "y_min", "x_max", "y_max",
    "x_center", "y_center",
    "speed_px_per_sec",
    "timestamp"
])

# =====================================
# 7) PROCESS FRAMES, COMPUTE SPEED & LOG
# =====================================
prev_x = None
prev_y = None
frame_idx = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference (returns a list; take first element)
        infer_start = time.time()
        results = model(rgb_frame, device=device_to_use, verbose=False)[0]
        infer_end = time.time()
        inference_time = infer_end - infer_start

        boxes = results.boxes  # all detected boxes

        # Default values if no detection
        xm = ym = None
        speed_px_per_sec = 0.0
        x_min = y_min = x_max = y_max = None

        if boxes:
            # Filter by confidence > 0.2
            filtered = [b for b in boxes if b.conf.item() > 0.2]
            if filtered:
                # Pick highest‐confidence box
                top = max(filtered, key=lambda b: b.conf.item())
                coords = top.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = coords
                x_min, y_min, x_max, y_max = map(int, (x1, y1, x2, y2))

                # Compute center
                xm = int((x_min + x_max) / 2)
                ym = int((y_min + y_max) / 2)

                # Compute speed if we have a previous center
                if prev_x is not None and prev_y is not None:
                    dx = xm - prev_x
                    dy = ym - prev_y
                    speed_px_per_sec = ((dx**2 + dy**2) ** 0.5) * fps

                prev_x, prev_y = xm, ym

                # Draw bounding box (green) and center marker (red)
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    thickness=2
                )
                cv2.circle(
                    frame,
                    (xm, ym),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1
                )

        # Compute timestamp for this frame
        time_offset_sec = frame_idx / fps
        timestamp = start_dt + timedelta(seconds=time_offset_sec)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Log only if we detected a valid center
        if xm is not None and ym is not None:
            writer.writerow([
                frame_idx,
                x_min, y_min, x_max, y_max,
                xm, ym,
                f"{speed_px_per_sec:.2f}",
                timestamp_str
            ])

        # Overlay Inference Time, Speed, and Timestamp on frame
        cv2.putText(
            frame,
            f"Infer: {inference_time:.3f}s ({device_to_use.upper()})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Speed: {speed_px_per_sec:.1f} px/s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            timestamp_str,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Display the frame
        cv2.imshow("YOLO + BBox + Speed + Time", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user (q pressed).")
            break

# =====================================
# 8) CLEAN UP & REPORT
# =====================================
cap.release()
cv2.destroyAllWindows()
f.close()

program_end_time = time.time()
program_duration = program_end_time - program_start_time

print("\nVideo processing complete.")
print(f"Processed {frame_idx} frames.")
print(f"Trajectory (with bbox, speed & timestamp) saved to: '{csv_path}'")
print(f"Program ended at:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(program_end_time))}")
print(f"Total execution time: {program_duration:.2f} seconds")
