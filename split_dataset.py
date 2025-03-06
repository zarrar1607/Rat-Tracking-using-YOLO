import os
import shutil
import random
from collections import defaultdict

def split_dataset_grouped_by_video(
    annotations_dir="annotations",
    output_dir="new_dataset",
    train_ratio=0.75,
    val_ratio=0.20,
    test_ratio=0.05,
    seed=42
):
    """
    Splits the images and labels into train, val, and test sets
    according to the specified ratios, but keeps each video’s frames
    together in the same proportion.

    Expects filenames like:
      VideoTitle_frame_00001.jpg
      VideoTitle_frame_00002.jpg
      ...

    Directory structure of annotations_dir:
      annotations_dir/
      ├── images
      │   ├── Baseline_frame_00001.jpg
      │   ├── Baseline2_frame_00369.jpg
      │   └── ...
      └── labels
          ├── Baseline_frame_00001.txt
          ├── Baseline2_frame_00369.txt
          └── ...

    Resulting structure in output_dir:
      new_dataset/
      ├── train
      │   ├── images
      │   └── labels
      ├── valid
      │   ├── images
      │   └── labels
      └── test
          ├── images
          └── labels
    """

    # Validate that ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Create output folders
    subsets = ["train", "valid", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)

    # Paths to the images and labels folders
    images_dir = os.path.join(annotations_dir, "images")
    labels_dir = os.path.join(annotations_dir, "labels")

    # 1. Group images by their video title
    #    We assume filenames like "VideoTitle_frame_XXXXX.jpg"
    video_groups = defaultdict(list)

    all_files = os.listdir(images_dir)
    for filename in all_files:
        # Filter image files only
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Extract the "video title" from the filename
        # Example: "Baseline2_frame_00369.jpg" -> video_title = "Baseline2"
        base, ext = os.path.splitext(filename)
        parts = base.split("_frame_")  # split at "_frame_"
        if len(parts) > 1:
            video_title = parts[0]
        else:
            # fallback if no "_frame_" is found
            video_title = base

        video_groups[video_title].append(filename)

    # 2. For each video group, shuffle and split into train/val/test
    random.seed(seed)

    def copy_image_and_label(img_file, subset):
        # Copy image
        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_dir, subset, "images", img_file)
        shutil.copy2(src_img_path, dst_img_path)

        # Copy label (if exists)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label_path = os.path.join(labels_dir, label_file)
        dst_label_path = os.path.join(output_dir, subset, "labels", label_file)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

    total_images = 0
    train_count = 0
    val_count = 0
    test_count = 0

    for video_title, file_list in video_groups.items():
        # Shuffle the frames of this video
        random.shuffle(file_list)
        n = len(file_list)
        total_images += n

        # Compute split indices
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train_files = file_list[:train_end]
        val_files = file_list[train_end:val_end]
        test_files = file_list[val_end:]

        train_count += len(train_files)
        val_count += len(val_files)
        test_count += len(test_files)

        # Copy each subset
        for f in train_files:
            copy_image_and_label(f, "train")
        for f in val_files:
            copy_image_and_label(f, "valid")
        for f in test_files:
            copy_image_and_label(f, "test")

    print(f"Total images: {total_images}")
    print(f"Train: {train_count} images")
    print(f"Val:   {val_count} images")
    print(f"Test:  {test_count} images")


if __name__ == "__main__":
    split_dataset_grouped_by_video(
        annotations_dir="annotations",
        output_dir="dataset",
        train_ratio=0.75,
        val_ratio=0.20,
        test_ratio=0.05,
        seed=42
    )
