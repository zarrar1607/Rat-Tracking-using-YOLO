import os
import cv2
import glob

def read_annotations(label_path, img_width, img_height):
    """
    Reads YOLO-format annotations from a file and converts them to absolute pixel coordinates.
    YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
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
            # Convert normalized coordinates to absolute pixel values
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
    Reads annotations from the given label file and draws them on a copy of the frame.
    """
    h, w = frame.shape[:2]
    boxes = read_annotations(label_path, w, h)
    disp = frame.copy()
    for (x_min, y_min, x_max, y_max, cid) in boxes:
        cv2.rectangle(disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(disp, f"ID: {cid}", (x_min, max(y_min-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return disp

def main():
    # Base folder "Cohort_1" contains two subfolders: "images" and "labels"
    base_folder = "random_youtube_video"
    images_dir = os.path.join(base_folder, "images")
    labels_dir = os.path.join(base_folder, "labels")

    # Get list of all .jpg images (adjust extension if needed)
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))

    if not image_paths:
        print("No images found in", images_dir)
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Unable to load image: {img_path}")
            continue

        # Construct the corresponding label file path
        base_name = os.path.basename(img_path)
        label_name = os.path.splitext(base_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        disp = update_display(img, label_path)

        cv2.imshow("Label Check", disp)
        print(f"Reviewing: {img_path} with label: {label_path}")
        print("Press 'q' to quit or any other key to continue...")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
