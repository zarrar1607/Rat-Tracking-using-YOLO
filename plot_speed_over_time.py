import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- USER CONFIGURATION ---
# Path to your CSV and video file
csv_path = "trajectory_full.csv"
video_path = "Video/New_8229-3-4-25_Sleep Deprivation_video.mp4"

# Specify the start datetime as a string
# Example format: "2025-03-04 12:00:00"
start_datetime_str = "2025-03-04 12:00:00"

# Parse the start datetime
start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")

# --- READ CSV DATA ---
df = pd.read_csv(csv_path)
if df.empty:
    raise ValueError(f"No data found in {csv_path}")

# --- GET FPS FROM VIDEO ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# --- COMPUTE SPEED PER FRAME ---
# Calculate differences between consecutive frames
df["dx"] = df["x_center"].diff()
df["dy"] = df["y_center"].diff()
# Speed in px/sec = sqrt(dx^2 + dy^2) * fps
df["speed"] = (df[["dx", "dy"]].pow(2).sum(axis=1).pow(0.5)) * fps

# --- ASSIGN TIMESTAMPS ---
# Compute time offset (in seconds) for each frame
df["time_offset"] = df["frame"] / fps
# Create actual timestamp for each detection
df["timestamp"] = df["time_offset"].apply(lambda t: start_dt + timedelta(seconds=t))

# --- RESAMPLE TO 1-SECOND BINS ---
# Set timestamp as index
df.set_index("timestamp", inplace=True)

# Average speed per second (drops NaN speeds for first row)
speed_per_second = df["speed"].resample("1S").mean().fillna(0)

# --- PLOT SPEED OVER TIME ---
plt.figure(figsize=(12, 6))
plt.plot(speed_per_second.index, speed_per_second.values, linestyle="-", marker="", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Speed (px/sec)")
plt.title("Object Speed Over Time")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

