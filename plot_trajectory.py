import cv2
import time # Not used in this plotting-focused script but kept from original
import torch # Not used in this plotting-focused script but kept from original
from ultralytics import YOLO # Not used in this plotting-focused script but kept from original
import csv # Not used in this plotting-focused script but kept from original
import pandas as pd
import matplotlib.pyplot as plt
import os # For basename in title

# → Load your model (Not strictly needed for plotting existing CSV, but part of your original context)
# model_path = "runs/detect/train30/weights/best.pt"
# if os.path.exists(model_path):
#     model = YOLO(model_path)
# else:
#     print(f"Warning: Model path '{model_path}' not found. Proceeding with plotting only.")

# → Open your video (to get frame dimensions, if not hardcoded)
# input_video_path = "Video/Feedback.mp4" # Make sure this path is correct
input_video_path = "Video/New_8229-3-4-25_Sleep Deprivation_video.mp4"
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    # Try to get frame dimensions from a known source if video fails
    # For demonstration, let's assume some defaults if video can't be opened
    # but ideally, you'd have these stored or ensure the video path is always valid
    print(f"Warning: Could not open video {input_video_path}. Using default frame dimensions (1920x1080).")
    frame_w, frame_h = 1920, 1080 # Example default
else:
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release() # Release the capture object as we only needed dimensions

print(f"Using frame dimensions: Width={frame_w}, Height={frame_h}")

# → Frame center (not used in current plots but kept from original)
# cx0, cy0 = frame_w/2, frame_h/2

# → Load data from CSV
# csv_path = "Feedback Deprivation - Full File - Rat 2- 4-25-25 Start_video.csv" # Make sure this CSV exists and has data
csv_path = "Trajectory data New 8229 - Rat 2 - 3-4-25 START_2025-03-04_Sleep Deprivation.csv" # Make sure this CSV exists and has data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}. Please run the detection script first.")

df = pd.read_csv(csv_path)

if df.empty:
    print(f"The CSV file '{csv_path}' is empty. Cannot generate plots.")
    exit()

# --- Plot 1: Trajectory (Your existing code) ---
plt.figure(figsize=(10, 8)) # You can adjust figure size
# plot the trajectory
plt.plot(df["x_center"], df["y_center"],
         "-", marker=".", markersize=4, alpha=0.7, label="Trajectory")
# mark the origin (top-left of the frame)
# plt.scatter([0], [0],
#             color="red", marker="x", s=80, label="Origin (0,0 - Top-Left)")

# Mark the frame boundaries if desired
# plt.plot([0, frame_w, frame_w, 0, 0], [0, 0, frame_h, frame_h, 0], 'k--', alpha=0.5, label="Frame Boundary")


plt.gca().invert_yaxis()            # y increases downward
plt.xlabel("x coordinate (pixels from left)")
plt.ylabel("y coordinate (pixels from top)")
plt.title(f"Object Trajectory ({os.path.basename(input_video_path)})")
plt.legend(loc="best") # Changed to "best" or "upper right"
plt.xlim(0, frame_w)
plt.ylim(frame_h, 0) # Inverted y-axis means higher value is at the bottom of plot
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal') # Enforce aspect ratio based on data ranges
plt.tight_layout()
plt.show()


# --- Plot 2: Heatmap ---
plt.figure(figsize=(10, 8)) # New figure for the heatmap

# Number of bins for the 2D histogram. Adjust for desired granularity.
# More bins = finer detail but might be sparse if few data points.
# Fewer bins = coarser detail but better for visualizing general areas.
num_bins_x = frame_w // 20  # Example: 20 pixel wide bins
num_bins_y = frame_h // 20  # Example: 20 pixel tall bins
# Or fixed number of bins:
# num_bins_x = 50
# num_bins_y = 50


# Create the 2D histogram / heatmap
# counts: The 2D array of bin counts
# xedges, yedges: The bin edges
# image: The QuadMesh object returned by hist2d (we can use it or ignore it)
counts, xedges, yedges, image = plt.hist2d(
    df["x_center"],
    df["y_center"],
    bins=[num_bins_x, num_bins_y],  # Can be a single int for square bins or [binsx, binsy]
    range=[[0, frame_w], [0, frame_h]], # Ensure heatmap covers the whole frame
    cmap='hot'  # Colormap (e.g., 'hot', 'viridis', 'plasma', 'inferno', 'magma')
    # cmin=1 # Optional: set minimum count to display a color (useful to hide empty bins)
)

plt.colorbar(label='Detection Frequency')
plt.gca().invert_yaxis() # Match image coordinates (y increases downwards)
plt.xlabel("x coordinate (pixels from left)")
plt.ylabel("y coordinate (pixels from top)")
plt.title(f"Detection Heatmap ({os.path.basename(input_video_path)})")
plt.xlim(0, frame_w)
plt.ylim(frame_h, 0) # Inverted y-axis
# plt.axis('equal') # Optional: if you want the aspect ratio of bins to be square based on data
plt.gca().set_aspect('equal', adjustable='box') # Better for ensuring visual squareness
plt.tight_layout()
plt.show()

print("Plotting complete.")