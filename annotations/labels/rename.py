import os
import re

# Folder containing the files
folder_path = r"./"

# Regex to match filenames like "desk7.png" or "desk12.jpg"
pattern = re.compile(r"^desk(\d+)(\..+)$")  
# group(1) = number, group(2) = extension

# Go into that directory
os.chdir(folder_path)

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        number_str = match.group(1)        # e.g. "7", "12"
        ext = match.group(2)              # e.g. ".png", ".jpg"
        number_int = int(number_str)       # Convert to int
        # Zero-pad to 5 digits (change 5 to however many digits you want)
        new_filename = f"desk_frame_{number_int:05d}.txt"
        
        # Rename
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        
        print(f"Renamed: {filename} → {new_filename}")

print("✅ Renaming complete!")
