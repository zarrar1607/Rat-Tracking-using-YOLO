import os
import shutil
import random
from collections import defaultdict

def extract_video_group(filename):
    """
    Extract a video group identifier from the filename.
    Uses the part before "_frame_" if available;
    otherwise, uses the first part before an underscore.
    """
    base, _ = os.path.splitext(filename)
    if "_frame_" in base:
        return base.split("_frame_")[0]
    elif "_" in base:
        return base.split("_")[0]
    else:
        return base

def extract_frame_number(filename):
    """
    Extracts a frame number from a filename.
    If the filename contains "_frame_", uses the part after it.
    Otherwise, if there is an underscore, tries to convert the last part.
    """
    base, _ = os.path.splitext(filename)
    if "_frame_" in base:
        parts = base.split("_frame_")
        try:
            return int(parts[1])
        except ValueError:
            return 0
    elif "_" in base:
        parts = base.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    else:
        return 0

def split_dataset_grouped_by_video(
    annotations_dir="annotations",
    output_dir="new_dataset",
    train_ratio=0.7,
    val_ratio=0.3,
    test_ratio=0.0,
    seed=42
):
    """
    Splits images and labels into train, valid, and test sets.
    
    Files are expected to be named like:
      VideoTitle_frame_XXXXX.jpg
      or other similar formats.
    
    For each video group (all files sharing the same video identifier),
    the frames are randomly shuffled and then split into
    train, valid, and test according to the specified ratios.
    
    This ensures that frames from each video are divided randomly
    according to the desired percentages.
    """
    # Validate that ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Create output folders for each subset
    for subset in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)

    images_dir = os.path.join(annotations_dir, "images")
    labels_dir = os.path.join(annotations_dir, "labels")

    # Group images by video group using our extraction function
    video_groups = defaultdict(list)
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        group = extract_video_group(filename)
        video_groups[group].append(filename)

    total_images = 0
    overall_counts = {"train": 0, "valid": 0, "test": 0}

    def copy_image_and_label(img_file, subset):
        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_dir, subset, "images", img_file)
        shutil.copy2(src_img_path, dst_img_path)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label_path = os.path.join(labels_dir, label_file)
        dst_label_path = os.path.join(output_dir, subset, "labels", label_file)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

    random.seed(seed)
    for group, file_list in video_groups.items():
        # Randomize the order of frames for a random split
        random.shuffle(file_list)
        n = len(file_list)
        total_images += n

        # Compute counts for each subset
        n_train = round(n * train_ratio)
        n_valid = round(n * val_ratio)
        n_test = n - n_train - n_valid  # Assign the remainder to test

        train_files = file_list[:n_train]
        valid_files = file_list[n_train:n_train+n_valid]
        test_files = file_list[n_train+n_valid:]

        overall_counts["train"] += len(train_files)
        overall_counts["valid"] += len(valid_files)
        overall_counts["test"] += len(test_files)

        for f in train_files:
            copy_image_and_label(f, "train")
        for f in valid_files:
            copy_image_and_label(f, "valid")
        for f in test_files:
            copy_image_and_label(f, "test")

    print(f"Total images: {total_images}")
    print(f"Train: {overall_counts['train']} images")
    print(f"Valid: {overall_counts['valid']} images")
    print(f"Test: {overall_counts['test']} images")

if __name__ == "__main__":
    # Example usage:
    split_dataset_grouped_by_video(
        annotations_dir="annotations",
        output_dir="dataset",
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        seed=62
    )
