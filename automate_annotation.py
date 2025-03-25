import os
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# (Optional) Force YOLO to run on CPU only:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# torch.set_num_threads(1)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
model_path = "runs/detect/train19/weights/best.pt"  # YOLOv8 model path
input_video_path = "Video/3_mice.mp4"
max_detections = 3           # How many top detections to store per frame
class_id = 0                 # Class ID to use in label file (e.g., '0' for rat)
confidence_threshold = 0.25  # Optional confidence threshold (model.predict(conf=...))

# ------------------------------------------------------------------------------
# Setup output folder structure
# ------------------------------------------------------------------------------
video_basename = os.path.basename(input_video_path)   # e.g. "3_mice.mp4"
video_title, _ = os.path.splitext(video_basename)     # e.g. "3_mice"

output_dir = video_title
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load the YOLOv8 model
# ------------------------------------------------------------------------------
model = YOLO(model_path)

# ------------------------------------------------------------------------------
# Open the video file
# ------------------------------------------------------------------------------
cap = cv2.VideoCapture(input_video_path)
frame_count = 0
start_time_global = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video or cannot read frame

    # ----------------------------------------------------------------------------
    # Run inference
    # ----------------------------------------------------------------------------
    start_infer = time.time()
    results = model.predict(frame, conf=confidence_threshold)
    end_infer = time.time()
    inference_time = end_infer - start_infer

    # ----------------------------------------------------------------------------
    # Process detections
    # ----------------------------------------------------------------------------
    detection_result = results[0]
    annotated_frame = frame.copy()

    label_lines = []  # lines to write to the YOLO label file
    if len(detection_result.boxes) > 0:
        # Extract confidence scores
        conf_all = detection_result.boxes.conf.cpu().numpy()  # shape (N,)
        # Sort in descending order
        sorted_indices = np.argsort(conf_all)[::-1]
        # Keep only top N
        top_n = min(max_detections, len(sorted_indices))
        top_indices = sorted_indices[:top_n]

        # Extract xywh in CPU numpy
        boxes_xywh_all = detection_result.boxes.xywh.cpu().numpy()  # shape (N,4)

        # Draw + compute YOLO-format for each top detection
        h_img, w_img = frame.shape[:2]
        for idx in top_indices:
            cx, cy, w_box, h_box = boxes_xywh_all[idx]
            x1 = int(cx - w_box / 2)
            y1 = int(cy - h_box / 2)
            x2 = int(cx + w_box / 2)
            y2 = int(cy + h_box / 2)

            # Draw bounding box on annotated_frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            score = conf_all[idx]
            label_text = f"Score: {score:.2f}"
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # ---------------------------------------------------------
            # Convert to YOLO format:
            # class, x_center_norm, y_center_norm, width_norm, height_norm
            # ---------------------------------------------------------
            x_center_norm = cx / w_img
            y_center_norm = cy / h_img
            w_norm = w_box / w_img
            h_norm = h_box / h_img

            # Round or format as needed
            line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            label_lines.append(line)

    # ----------------------------------------------------------------------------
    # Save the image and label
    # ----------------------------------------------------------------------------
    # 1) Save the original image for training (uncomment if you want annotated version)
    image_filename = f"{video_title}_frame_{frame_count:06d}.jpg"
    image_path = os.path.join(images_dir, image_filename)
    cv2.imwrite(image_path, frame)  # or use annotated_frame if desired

    # 2) Write label file
    label_filename = f"{video_title}_frame_{frame_count:06d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, "w") as f:
        for line in label_lines:
            f.write(line + "\n")

    frame_count += 1

    # ----------------------------------------------------------------------------
    # Display the annotated frame (optional)
    # ----------------------------------------------------------------------------
    cv2.putText(
        annotated_frame,
        f"Inference Time: {inference_time:.3f} sec",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting display loop by user command.")
        break

cap.release()
cv2.destroyAllWindows()

print("Video processing complete. Display closed.")
print(f"Saved {frame_count} frames and label files to '{output_dir}/images' and '{output_dir}/labels'.")
