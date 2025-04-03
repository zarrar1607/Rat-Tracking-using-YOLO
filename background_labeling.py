import cv2
import os

base_name = 'desktop'

video_path = f'Video/{base_name}.avi'
output_img_dir = f'{base_name}/images'
output_lbl_dir = f'{base_name}/labels'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # File names like: destop_0000.jpg and destop_0000.txt
    img_name = f"{base_name}_frame_{frame_idx:04d}.jpg"
    label_name = f"{base_name}_frame_{frame_idx:04d}.txt"

    # Save image
    cv2.imwrite(os.path.join(output_img_dir, img_name), frame)

    # Create empty label file (negative sample)
    with open(os.path.join(output_lbl_dir, label_name), 'w') as f:
        pass

    frame_idx += 1

cap.release()
print(f"✅ Done extracting frames from '{video_path}' into '{base_name}/train/'")
