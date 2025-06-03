import cv2
import pandas as pd
import os
from datetime import datetime, timedelta

# # ======================================================
# # 1) USER CONFIGURATION: paths, start datetime, etc.
# # ======================================================

# # Path to your existing trajectory CSV (must have columns: frame, x_center, y_center)
# input_csv = "Trajectory data New 8229 - Rat 2 - 3-4-25 START_2025-03-04_Sleep Deprivation.csv"
# if not os.path.exists(input_csv):
#     raise FileNotFoundError(f"CSV file not found: {input_csv}")

# # Path to the original video (used only to obtain FPS)
# video_path = "Video/Feedback.mp4"
# if not os.path.exists(video_path):
#     raise FileNotFoundError(f"Video file not found: {video_path}")

# # Start datetime corresponding to frame 0 of the video (format: "YYYY-MM-DD HH:MM:SS")
# start_datetime_str = "2025-03-06 06:00:00"
# start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")

# # Path to the output CSV that will include speed & timestamp
# output_csv = "Trajectory trajectory_with_speed_timestamp.csv"

# # ======================================================
# # 2) READ INPUT CSV INTO PANDAS
# # ======================================================

# df = pd.read_csv(input_csv)
# required_cols = {"frame", "x_center", "y_center"}
# if not required_cols.issubset(df.columns):
#     raise ValueError(f"Input CSV must contain columns: {required_cols}")

# # Ensure frame is integer
# df["frame"] = df["frame"].astype(int)

# # Sort by frame (in case it’s not already sorted)
# df = df.sort_values("frame").reset_index(drop=True)

# # ======================================================
# # 3) OPEN VIDEO TO GET FPS (NO DISPLAY)
# # ======================================================

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise RuntimeError(f"Could not open video file: {video_path}")

# fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# cap.release()
# print(f"Loaded video '{video_path}' → FPS = {fps:.2f}")

# # ======================================================
# # 4) COMPUTE SPEED & TIMESTAMP FOR EACH ROW
# # ======================================================

# # We'll build lists to hold our new columns
# timestamps = []
# speeds_px_s = []

# # Keep track of the previous point's frame and coordinates
# prev_frame = None
# prev_x = None
# prev_y = None

# for idx, row in df.iterrows():
#     frame_i = row["frame"]
#     x_i     = float(row["x_center"])
#     y_i     = float(row["y_center"])

#     # 4a) Compute timestamp: start_dt + (frame / fps) seconds
#     elapsed_sec = frame_i / fps
#     ts = start_dt + timedelta(seconds=elapsed_sec)
#     timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))

#     # 4b) Compute speed in px/sec
#     if prev_frame is None:
#         # No previous point → speed = 0
#         speed = 0.0
#     else:
#         # time difference in seconds between current frame and previous frame
#         dt = (frame_i - prev_frame) / fps
#         if dt <= 0:
#             speed = 0.0
#         else:
#             dist = ((x_i - prev_x) ** 2 + (y_i - prev_y) ** 2) ** 0.5
#             speed = dist / dt  # px per second

#     speeds_px_s.append(speed)

#     # update previous values
#     prev_frame = frame_i
#     prev_x = x_i
#     prev_y = y_i

# # ======================================================
# # 5) ADD COLUMNS TO DATAFRAME AND WRITE OUTPUT CSV
# # ======================================================

# df["speed_px_per_sec"] = speeds_px_s
# df["timestamp"]        = timestamps

# # Reorder columns if desired
# df_out = df[["frame", "x_center", "y_center", "speed_px_per_sec", "timestamp"]]

# df_out.to_csv(output_csv, index=False)
# print(f"Written output to '{output_csv}' ({len(df_out)} rows).")


# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # 1) Load your CSV as before
# # csv_path = "Feedback Deprivation trajectory_with_speed_timestamp.csv"
# csv_path = "Feedback Deprivation trajectory_with_speed_timestamp.csv"
# df = pd.read_csv(csv_path)
# df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

# # 2) Resample to 1‐second median (as in previous steps)
# df.set_index("timestamp", inplace=True)
# median_speed_per_sec = (
#     df["speed_px_per_sec"]
#       .fillna(0)
# )

# # 3) Plot, then set a custom DateFormatter for the x‐axis
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(median_speed_per_sec.index, median_speed_per_sec.values, linewidth=1)

# # Tell Matplotlib to use our exact format for every tick
# ax.xaxis.set_major_formatter(
#     mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
# )

# # Optionally, rotate labels so they don’t overlap
# plt.xticks(rotation=45, ha="right")

# ax.set_xlabel("Time (YYYY-MM-DD HH:MM:SS)")
# ax.set_ylabel(" Speed (px/sec)")
# ax.set_title("Object Speed  vs. Time")
# ax.grid(True, linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1) Load the CSV and parse timestamps
csv_path = "Feedback Deprivation trajectory_with_speed_timestamp.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
df.set_index("timestamp", inplace=True)

# 2) Extract per‐frame speed series (fill missing as 0)
per_second_speed = df["speed_px_per_sec"].fillna(0)

# 3) Compute hourly average speed (mean and median)
hourly_avg_speed_mean   = df["speed_px_per_sec"].resample("1H").mean().fillna(0)
hourly_avg_speed_median = df["speed_px_per_sec"].resample("1H").median().fillna(0)

# 4) Create a 2×2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharey=False)

# ——————————————————————————————————————————————
# Top‐left: per‐second speed vs. timestamp (ax = axs[0,0])
# ——————————————————————————————————————————————
ax1 = axs[0, 0]
ax1.plot(per_second_speed.index, per_second_speed.values, linewidth=1, color="tab:blue")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
ax1.set_xlabel("Time (YYYY-MM-DD HH:MM:SS)")
ax1.set_ylabel("Speed (px/sec)")
ax1.set_title("Object Speed vs. Time (Per Frame)")
ax1.grid(True, linestyle="--", alpha=0.4)
for label in ax1.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Top‐right: hourly mean speed vs. hour (ax = axs[0,1])
# ——————————————————————————————————————————————
ax2 = axs[0, 1]
ax2.plot(hourly_avg_speed_mean.index, hourly_avg_speed_mean.values,
         marker="o", linestyle="-", color="tab:orange")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
ax2.set_xlabel("Time (Hourly)")
ax2.set_ylabel("Mean Speed (px/sec)")
ax2.set_title("Hourly Mean Object Speed")
ax2.grid(True, linestyle="--", alpha=0.4)
for label in ax2.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Bottom‐left: hourly median speed vs. hour (ax = axs[1,0])
# ——————————————————————————————————————————————
ax3 = axs[1, 0]
ax3.plot(hourly_avg_speed_median.index, hourly_avg_speed_median.values,
         marker="o", linestyle="-", color="tab:green")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
ax3.set_xlabel("Time (Hourly)")
ax3.set_ylabel("Median Speed (px/sec)")
ax3.set_title("Hourly Median Object Speed")
ax3.grid(True, linestyle="--", alpha=0.4)
for label in ax3.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Bottom‐right: empty (ax = axs[1,1]) – placeholder or remove if not needed
# ——————————————————————————————————————————————
ax4 = axs[1, 1]
ax4.axis("off")  # turn off this subplot entirely

plt.title("Feedback Deprivation - Full File - Rat 2- 4-25-25 Start")
# ——————————————————————————————————————————————
# 5) TIGHT LAYOUT & SHOW
# ——————————————————————————————————————————————
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1) Load the CSV and parse timestamps
csv_path = "Trajectory trajectory_with_speed_timestamp.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
df.set_index("timestamp", inplace=True)

# 2) Extract per‐frame speed series (fill missing as 0)
per_second_speed = df["speed_px_per_sec"].fillna(0)

# 3) Compute hourly average speed (mean and median)
hourly_avg_speed_mean   = df["speed_px_per_sec"].resample("1H").mean().fillna(0)
hourly_avg_speed_median = df["speed_px_per_sec"].resample("1H").median().fillna(0)

# 4) Create a 2×2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharey=False)

# ——————————————————————————————————————————————
# Top‐left: per‐second speed vs. timestamp (ax = axs[0,0])
# ——————————————————————————————————————————————
ax1 = axs[0, 0]
ax1.plot(per_second_speed.index, per_second_speed.values, linewidth=1, color="tab:blue")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
ax1.set_xlabel("Time (YYYY-MM-DD HH:MM:SS)")
ax1.set_ylabel("Speed (px/sec)")
ax1.set_title("Object Speed vs. Time (Per Frame)")
ax1.grid(True, linestyle="--", alpha=0.4)
for label in ax1.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Top‐right: hourly mean speed vs. hour (ax = axs[0,1])
# ——————————————————————————————————————————————
ax2 = axs[0, 1]
ax2.plot(hourly_avg_speed_mean.index, hourly_avg_speed_mean.values,
         marker="o", linestyle="-", color="tab:orange")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
ax2.set_xlabel("Time (Hourly)")
ax2.set_ylabel("Mean Speed (px/sec)")
ax2.set_title("Hourly Mean Object Speed")
ax2.grid(True, linestyle="--", alpha=0.4)
for label in ax2.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Bottom‐left: hourly median speed vs. hour (ax = axs[1,0])
# ——————————————————————————————————————————————
ax3 = axs[1, 0]
ax3.plot(hourly_avg_speed_median.index, hourly_avg_speed_median.values,
         marker="o", linestyle="-", color="tab:green")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
ax3.set_xlabel("Time (Hourly)")
ax3.set_ylabel("Median Speed (px/sec)")
ax3.set_title("Hourly Median Object Speed")
ax3.grid(True, linestyle="--", alpha=0.4)
for label in ax3.get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

# ——————————————————————————————————————————————
# Bottom‐right: empty (ax = axs[1,1]) – placeholder or remove if not needed
# ——————————————————————————————————————————————
ax4 = axs[1, 1]
ax4.axis("off")  # turn off this subplot entirely

plt.title("New 8229 - Rat 2 - 3-4-25 START_2025-03-04_Sleep Deprivation")
# ——————————————————————————————————————————————
# 5) TIGHT LAYOUT & SHOW
# ——————————————————————————————————————————————
plt.tight_layout()
plt.show()
