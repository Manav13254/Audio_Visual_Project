import os
import shutil
import glob
import random

RANDOM_SEED = 1154
SPLIT_RATIO = 0.8   # 80% train, 20% test

VISION_SRC = r"E:\Audio_Visual_Project\Advance dataset\ADVANCE_vision\vision"
SOUND_SRC  = r"E:\Audio_Visual_Project\Advance dataset\ADVANCE_sound\sound"
OUT_ROOT   = r"E:\Audio_Visual_Project\ADVANCE_DATA_split"

for mode in ['vision', 'sound']:
    src_root = VISION_SRC if mode == 'vision' else SOUND_SRC
    for cls_name in sorted(os.listdir(src_root)):
        cls_path = os.path.join(src_root, cls_name)
        if not os.path.isdir(cls_path): continue
        files = sorted(glob.glob(os.path.join(cls_path, "*")))
        random.Random(RANDOM_SEED).shuffle(files)
        split_idx = int(SPLIT_RATIO * len(files))
        train_files, test_files = files[:split_idx], files[split_idx:]

        for split, split_files in zip(['train', 'test'], [train_files, test_files]):
            out_dir = os.path.join(OUT_ROOT, split, mode, cls_name)
            os.makedirs(out_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir)

        print(f"{mode}/{cls_name}: {len(train_files)} train, {len(test_files)} test")

print("DONE splitting in ADVANCE_split/[train|test]/[vision|sound]/<class>")
