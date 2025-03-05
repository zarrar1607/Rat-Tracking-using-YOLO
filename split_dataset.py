import os
import shutil
import random

def split_dataset(
    annotations_dir="annotations",
    output_dir="new_dataset",
    train_ratio=0.80,
    val_ratio=0.15,
    test_ratio=0.05,
    seed=42
):
    """
    Splits the images and labels into train, val, and test sets
    according to the specified ratios.
    
    :param annotations_dir: Path to the directory containing 'images' and 'labels' folders.
    :param output_dir: Path where 'train', 'val', and 'test' folders will be created (default: 'mew_Dataset').
    :param train_ratio: Proportion of data to go into the 'train' set.
    :param val_ratio: Proportion of data to go into the 'val' set.
    :param test_ratio: Proportion of data to go into the 'test' set.
    :param seed: Random seed for reproducible shuffling.
    """

    # --- Validate Ratios ---
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # --- Create Output Folders ---
    subsets = ["train", "val", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)

    # --- Collect all image filenames ---
    images_dir = os.path.join(annotations_dir, "images")
    labels_dir = os.path.join(annotations_dir, "labels")
    
    all_images = [
        f for f in os.listdir(images_dir) 
        if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    all_images.sort()  # optional sorting

    # --- Shuffle data for random splitting ---
    random.seed(seed)
    random.shuffle(all_images)

    # --- Calculate split indices ---
    total_images = len(all_images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]

    # --- Helper function to copy image+label to subset folder ---
    def copy_to_subset(file_list, subset_name):
        for img_file in file_list:
            # Copy image
            src_img_path = os.path.join(images_dir, img_file)
            dst_img_path = os.path.join(output_dir, subset_name, "images", img_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy label (if it exists)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(labels_dir, label_file)
            dst_label_path = os.path.join(output_dir, subset_name, "labels", label_file)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)

    # --- Copy files into respective folders ---
    copy_to_subset(train_files, "train")
    copy_to_subset(val_files, "valid")
    copy_to_subset(test_files, "test")

    print(f"Total images: {total_images}")
    print(f"Train: {len(train_files)} images")
    print(f"Val:   {len(val_files)} images")
    print(f"Test:  {len(test_files)} images")

if __name__ == "__main__":
    split_dataset(
        annotations_dir="annotations",  # Path to your 'annotations' folder
        output_dir="new_dataset",       # New folder to store train/val/test splits
        train_ratio=0.80,
        val_ratio=0.15,
        test_ratio=0.05,
        seed=42
    )
