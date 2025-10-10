import os
import shutil
import random
import math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

AUDIO_SOURCE_DIR = Path(r"E:\Audio_Visual_Project\Advance dataset\ADVANCE_sound\sound")
VISION_SOURCE_DIR = Path(r"E:\Audio_Visual_Project\Advance dataset\ADVANCE_vision\vision")
OUTPUT_DIR = Path(r"E:\Audio_Visual_Project\final_data_split_80_20")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def get_group_id(filename):
    return filename.stem.split('_')[0]

def split_data():
    print("--- Starting Dataset Split (80:20 Train/Val) ---")
    if not AUDIO_SOURCE_DIR.exists() or not VISION_SOURCE_DIR.exists():
        print("Error: Source directory not found. Please check the paths.")
        return

    groups_by_class = defaultdict(set)
    class_names = sorted([d.name for d in AUDIO_SOURCE_DIR.iterdir() if d.is_dir()])
    for class_name in class_names:
        class_audio_dir = AUDIO_SOURCE_DIR / class_name
        for audio_file in class_audio_dir.glob('*.wav'):
            group_id = get_group_id(audio_file)
            groups_by_class[class_name].add(group_id)

    split_groups = {'train': defaultdict(list), 'val': defaultdict(list)}
    for class_name, groups in groups_by_class.items():
        group_list = sorted(list(groups))
        random.shuffle(group_list)
        n_total = len(group_list)
        n_train = math.floor(n_total * TRAIN_RATIO)
        split_groups['train'][class_name] = group_list[:n_train]
        split_groups['val'][class_name] = group_list[n_train:]

    print("Copying files to destination folders...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_groups_to_copy = sum(len(ids) for cls in split_groups.values() for ids in cls.values())
    with tqdm(total=total_groups_to_copy, desc="Copying Groups") as pbar:
        for split_name, class_data in split_groups.items():
            for class_name, group_ids in class_data.items():
                dest_audio_dir = OUTPUT_DIR / split_name / 'audio' / class_name
                dest_vision_dir = OUTPUT_DIR / split_name / 'vision' / class_name
                
                dest_audio_dir.mkdir(parents=True, exist_ok=True)
                dest_vision_dir.mkdir(parents=True, exist_ok=True)

                for group_id in group_ids:
                    audio_files = list((AUDIO_SOURCE_DIR / class_name).glob(f"{group_id}*.wav"))
                    for file_path in audio_files:
                        shutil.copy(file_path, dest_audio_dir)
                    
                    vision_files = list((VISION_SOURCE_DIR / class_name).glob(f"{group_id}*.jpg"))
                    for file_path in vision_files:
                        shutil.copy(file_path, dest_vision_dir)
                
                pbar.update(len(group_ids))

    print("\n--- Split Verification ---")
    for split_name, class_data in split_groups.items():
        total_groups = sum(len(g) for g in class_data.values())
        print(f"Total groups in {split_name.upper()}: {total_groups}")
        
    print("\n✅ Dataset splitting complete!")

if __name__ == '__main__':
    if not math.isclose(TRAIN_RATIO + VAL_RATIO, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    split_data()

