import os
import cv2
import numpy as np

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
video_path = "Video/3_mice.mp4"
start_frame = 6500  # Frame number from which to start reviewing/annotating

# Output folders based on video filename (e.g., "3_mice")
video_basename = os.path.basename(video_path)   # e.g., "3_mice.mp4"
video_title, _ = os.path.splitext(video_basename) # e.g., "3_mice"
output_dir = video_title
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def read_annotations(label_path, img_width, img_height):
    """
    Reads YOLO-format annotations from a file and converts them to absolute pixel coordinates.
    YOLO format: class_id, x_center_norm, y_center_norm, width_norm, height_norm
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
            except Exception as e:
                print(f"Error parsing line in {label_path}: {line}")
                continue
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            box_width = width_norm * img_width
            box_height = height_norm * img_height
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)
            boxes.append((x_min, y_min, x_max, y_max, class_id))
    return boxes

def update_display(frame, label_path):
    """
    Updates the display image by reading annotations and drawing them (in green) on the frame.
    """
    h, w = frame.shape[:2]
    boxes = read_annotations(label_path, w, h)
    disp = frame.copy()
    for (x_min, y_min, x_max, y_max, cid) in boxes:
        cv2.rectangle(disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(disp, f"ID: {cid}", (x_min, max(y_min-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return disp

# Global variables for manual annotation.
drawing = False
start_pt = None
new_boxes = []   # List to store manually drawn boxes (x_min, y_min, x_max, y_max)
temp_img = None  # Temporary image for drawing updates

def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback function for manual drawing.
    Left-click and drag to draw a rectangle.
    """
    global drawing, start_pt, new_boxes, temp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_disp.copy()  # base image without new boxes drawn
            cv2.rectangle(temp_img, start_pt, (x, y), (0, 0, 255), 2)
            for box in new_boxes:
                cv2.rectangle(temp_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.imshow("Label Check", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_pt = (x, y)
        new_boxes.append((start_pt[0], start_pt[1], end_pt[0], end_pt[1]))
        temp_img = img_disp.copy()
        for box in new_boxes:
            cv2.rectangle(temp_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow("Label Check", temp_img)

# ------------------------------------------------------------------------------
# Open Video and Set Starting Frame
# ------------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_count = start_frame

# ------------------------------------------------------------------------------
# Main Loop for Reviewing/Annotating Frames
# ------------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read.")
        break

    # Generate filenames based on frame_count.
    image_filename = f"{video_title}_frame_{frame_count:06d}.jpg"
    image_path = os.path.join(images_dir, image_filename)
    label_filename = f"{video_title}_frame_{frame_count:06d}.txt"
    label_path = os.path.join(labels_dir, label_filename)

    # Save frame if not already saved.
    if not os.path.exists(image_path):
        cv2.imwrite(image_path, frame)

    h, w = frame.shape[:2]
    img_disp = update_display(frame, label_path)

    # Reset manual annotation variables.
    new_boxes = []
    temp_img = img_disp.copy()

    cv2.imshow("Label Check", img_disp)
    print(f"Reviewing frame {frame_count}.")
    print("Press 'd' to delete image and label file, 'c' to clear & manually annotate, 'n' for next, 'p' for previous, or 'q' to quit.")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        cv2.destroyWindow("Label Check")
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted label file: {label_path}")
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image file: {image_path}")
        frame_count += 1
    elif key == ord('c'):
        print("Entering manual annotation mode.")
        print("Draw new boxes with left mouse button.")
        print("Press 'r' to clear drawn boxes, 's' to save annotations, or 'q' to exit manual mode without saving.")
        cv2.setMouseCallback("Label Check", draw_rectangle)
        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('s'):
                label_lines = []
                for (x_min, y_min, x_max, y_max) in new_boxes:
                    cx = (x_min + x_max) / 2.0
                    cy = (y_min + y_max) / 2.0
                    w_box = x_max - x_min
                    h_box = y_max - y_min
                    x_center_norm = cx / w
                    y_center_norm = cy / h
                    w_norm = w_box / w
                    h_norm = h_box / h
                    line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    label_lines.append(line)
                with open(label_path, "w") as f:
                    for line in label_lines:
                        f.write(line + "\n")
                print(f"Saved new annotations to {label_path}")
                cv2.setMouseCallback("Label Check", lambda *args: None)
                break
            elif key2 == ord('r'):
                new_boxes = []
                temp_img = img_disp.copy()
                cv2.imshow("Label Check", temp_img)
                print("Cleared drawn boxes. Draw new boxes, then press 's' to save or 'q' to exit manual mode.")
            elif key2 == ord('q'):
                print("Exiting manual annotation mode without saving.")
                cv2.setMouseCallback("Label Check", lambda *args: None)
                break
            else:
                print("Unrecognized key in manual mode. Options: 'r' to clear, 's' to save, 'q' to exit.")
        frame_count += 1
    elif key == ord('n'):
        frame_count += 1
    elif key == ord('p'):
        if frame_count > start_frame:
            frame_count -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if ret:
                image_filename = f"{video_title}_frame_{frame_count:06d}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                label_filename = f"{video_title}_frame_{frame_count:06d}.txt"
                label_path = os.path.join(labels_dir, label_filename)
                if not os.path.exists(image_path):
                    cv2.imwrite(image_path, frame)
                img_disp = update_display(frame, label_path)
                cv2.imshow("Label Check", img_disp)
                print(f"Reviewing frame {frame_count}.")
            else:
                print("Failed to read previous frame.")
            continue  # Skip rest of loop to re-read frame properly
        else:
            print("Already at the first frame.")
    elif key == ord('q'):
        print("Exiting label review.")
        break
    else:
        print("Unrecognized key. Skipping to next frame.")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Video processing complete. Display closed.")
