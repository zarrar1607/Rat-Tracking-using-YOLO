import os
import cv2
import numpy as np

# Modify these paths as needed.
train_dir = "new_dataset/train"
images_dir = os.path.join(train_dir, "images")
labels_dir = os.path.join(train_dir, "labels")

# Get sorted list of image files (you can add more extensions if needed)
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def read_annotations(label_path, img_width, img_height):
    """
    Reads YOLO-format annotations from a file and converts them to absolute coordinates.
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
            # Convert normalized coordinates to absolute pixel values.
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

i = 0
while i < len(image_files):
    img_file = image_files[i]
    img_path = os.path.join(images_dir, img_file)
    # Assume the label file has the same base name but with .txt extension.
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)
    
    # Load the image using OpenCV.
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        i += 1
        continue
    height, width = img.shape[:2]
    
    # Read annotations and draw bounding boxes.
    boxes = read_annotations(label_path, width, height)
    img_disp = img.copy()
    for (x_min, y_min, x_max, y_max, class_id) in boxes:
        # Draw rectangle (green) and class_id above it.
        cv2.rectangle(img_disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_disp, f"ID: {class_id}", (x_min, max(y_min-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the image with annotations.
    cv2.imshow("Label Check", img_disp)
    print(f"Reviewing {img_file} [{i+1}/{len(image_files)}]. Press 'd' to delete label file, 's' to skip, 'p' for previous, or 'q' to quit.")
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted label file: {label_path}")
        i += 1  # move to next image
    elif key == ord('s'):
        i += 1  # simply skip to next image
    elif key == ord('p'):
        if i > 0:
            i -= 1  # go back to previous image
        else:
            print("Already at the first image.")
    elif key == ord('q'):
        print("Exiting label review.")
        break
    else:
        print("Unrecognized key. Skipping to next image.")
        i += 1

cv2.destroyAllWindows()
