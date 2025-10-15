import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def split_image_data():
    source_folder = Path("D:/machine learning/Audio_Visual_Project/vision") 
    destination_folder = Path("D:/machine learning/Audio_Visual_Project/ADVANCE_images_split") 
    split_ratios = {'train': 0.7, 'val': 0.1, 'test': 0.2} 
    random.seed(42)
    if not source_folder.exists():
        print(f"Error: Source folder not found at {source_folder}")
        print("Please update the 'source_folder' variable to your image dataset's location.")
        return
    if destination_folder.exists():
        print(f"Warning: Destination folder {destination_folder} already exists. It will be reused.")
    print("Starting image data split...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    class_dirs = [d for d in source_folder.iterdir() if d.is_dir()]

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        file_groups = defaultdict(list)
        
        all_images = []
        for ext in image_extensions:
            all_images.extend(class_dir.glob(ext))
            
        for image_file in all_images:
            base_name = image_file.stem.split('_')[0]
            file_groups[base_name].append(image_file)
            
        group_keys = list(file_groups.keys())
        random.shuffle(group_keys)
        
        num_groups = len(group_keys)
        if num_groups == 0:
            continue 
            
        train_end = int(num_groups * split_ratios['train'])
        val_end = train_end + int(num_groups * split_ratios['val'])
        
        splits = {
            'train': group_keys[:train_end],
            'val': group_keys[train_end:val_end],
            'test': group_keys[val_end:]
        }
        
        for split_name, group_list in splits.items():
            dest_path = destination_folder / split_name / class_name
            dest_path.mkdir(parents=True, exist_ok=True)
            
            for base_name in group_list:
                for image_file in file_groups[base_name]:
                    shutil.copy(image_file, dest_path / image_file.name)

    print("\nImage data splitting complete!")
    print(f"Split image data is now available in: {destination_folder}")

if __name__ == "__main__":
    split_image_data()