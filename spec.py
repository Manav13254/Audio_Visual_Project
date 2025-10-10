import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

RAW_DATA_DIR = Path(r"E:\Audio_Visual_Project\final_data_split_80_20")
FEATURES_OUTPUT_DIR = Path(r"E:\Audio_Visual_Project\preprocessed_features_80_20")

SAMPLE_RATE = 16000
N_FFT = 4096
HOP_LENGTH = 400
WIN_LENGTH = 1024
N_MELS = 64
N_TIME_STEPS = 400

def audio_extract(wav_file):
    wav, _ = librosa.load(wav_file, sr=SAMPLE_RATE)
    spec = librosa.core.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH,
                             win_length=WIN_LENGTH, window='hann',
                             center=True, pad_mode='constant')
    mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=SAMPLE_RATE,
                                         n_mels=N_MELS, fmax=8000)

    if mel.shape[1] < N_TIME_STEPS:
        pad_width = N_TIME_STEPS - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel = mel[:, :N_TIME_STEPS]

    logmel = librosa.core.power_to_db(mel)
    return logmel.T.astype('float32')

def main():
    print("--- Starting Offline Feature Extraction (80:20 Train/Val) ---")
    FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        source_dir = RAW_DATA_DIR / split / "audio"
        dest_dir = FEATURES_OUTPUT_DIR / split
        
        if not source_dir.exists():
            print(f"Warning: Skipping non-existent source directory: {source_dir}")
            continue

        audio_files = list(source_dir.rglob('*.wav'))
        
        for audio_path in tqdm(audio_files, desc=f"Extracting {split} features"):
            relative_path = audio_path.relative_to(source_dir)
            output_path = (dest_dir / relative_path).with_suffix('.npy')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            features = audio_extract(str(audio_path))
            np.save(output_path, features)

    print("\nCalculating normalization statistics from the training set...")
    train_features_dir = FEATURES_OUTPUT_DIR / "train"
    train_feature_files = list(train_features_dir.rglob('*.npy'))
    
    if not train_feature_files:
        raise FileNotFoundError("No training features found to calculate stats.")

    total_sum = np.zeros(N_MELS, dtype=np.float64)
    total_count = 0
    for feature_path in tqdm(train_feature_files, desc="Pass 1/2 (Mean)"):
        features = np.load(feature_path)
        total_sum += np.sum(features, axis=0)
        total_count += features.shape[0]
    mu = total_sum / total_count

    total_sum_sq_diff = np.zeros(N_MELS, dtype=np.float64)
    for feature_path in tqdm(train_feature_files, desc="Pass 2/2 (Std Dev)"):
        features = np.load(feature_path)
        total_sum_sq_diff += np.sum((features - mu) ** 2, axis=0)
    sigma = np.sqrt(total_sum_sq_diff / total_count)
    sigma[sigma == 0] = 1e-8

    normalizer_path = FEATURES_OUTPUT_DIR / "normalizer.npy"
    np.save(normalizer_path, [mu.astype('float32'), sigma.astype('float32')])
    
    print(f"\n✅ Normalization statistics saved to: {normalizer_path}")
    print("--- Pre-processing complete! ---")

if __name__ == '__main__':
    main()

