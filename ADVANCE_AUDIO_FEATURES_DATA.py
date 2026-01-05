import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# Configuration
SOUND_SRC = r"E:/Audio_Visual_Project/ADVANCE_DATA_split"
OUT_ROOT = r"E:/Audio_Visual_Project/ADVANCE_features"
os.makedirs(OUT_ROOT, exist_ok=True)
sr = 16000
n_mels = 64
fea_len = 400

splits = ["train", "test"]  # Add 'val' if you have it!

# 1. Extract log-mel function
def extract_logmel(path):
    wav, _ = librosa.load(path, sr=sr)
    spec = librosa.stft(wav, n_fft=4096, hop_length=400, win_length=1024,
                        window='hann', center=True, pad_mode='constant')
    mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=n_mels, fmax=8000)
    logmel = librosa.power_to_db(mel[:, :fea_len])
    return logmel.T.astype('float32')  # (400, 64)

# 2. Main extraction loop
for split in splits:
    split_dir = os.path.join(SOUND_SRC, split, "sound")
    all_features = []
    all_labels = []
    all_fileids = []
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    for cls in tqdm(class_names, desc=f"{split.upper()} classes"):
        cls_in = os.path.join(split_dir, cls)
        cls_files = sorted(glob.glob(os.path.join(cls_in, "*.wav")))
        out_cls_dir = os.path.join(OUT_ROOT, split, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        for wav_path in tqdm(cls_files, desc=f"{split}-{cls}", leave=False):
            fea = extract_logmel(wav_path)
            # Save per-file .npy for loader, could also save as array batch
            fname = os.path.splitext(os.path.basename(wav_path))[0]
            npy_path = os.path.join(out_cls_dir, fname + ".npy")
            np.save(npy_path, fea)
            all_features.append(fea)
            all_labels.append(class_to_idx[cls])
            all_fileids.append(fname)
    print(f"Finished split: {split}. Saved {len(all_features)} feature files.")

# 3. Optionally calculate and save normalization stats (use only train!)
train_features = []
for cls in class_names:
    cls_dir = os.path.join(OUT_ROOT, "train", cls)
    for npyfile in sorted(os.listdir(cls_dir)):
        arr = np.load(os.path.join(cls_dir, npyfile))
        train_features.append(arr)
train_features = np.array(train_features)
mu = np.mean(train_features, axis=0)
sigma = np.std(train_features, axis=0)
np.save(os.path.join(OUT_ROOT, "normalizer_train.npy"), [mu, sigma])
print("Saved train normalization stats.")

print("✅ All done! Feature .npy files are ready for each class/split in ADVANCE_features/train/CLASS/<file>.npy , etc.")
