# Filename: preprocess_audio.py (Corrected Version)

import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 400
N_MELS = 64
CLIP_DURATION_S = 10
EXPECTED_SHAPE = (N_MELS, 400) 

def process_audio_file(file_path: Path) -> np.ndarray:
    try:
        waveform, sr = librosa.load(file_path, sr=TARGET_SR, duration=CLIP_DURATION_S)
        if len(waveform) < TARGET_SR * CLIP_DURATION_S:
            waveform = librosa.util.fix_length(waveform, size=TARGET_SR * CLIP_DURATION_S)
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        if log_mel_spectrogram.shape[1] != EXPECTED_SHAPE[1]:
            log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=EXPECTED_SHAPE[1], axis=1)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    input_data_path = Path("./data/ADVANCE_split")  
    output_data_path = Path("./data/processed_audio")

    if not input_data_path.exists():
        print(f"Error: Input directory not found at {input_data_path}")
        print("Please run the split_data.py script first.")
        return

    all_spectrograms = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing '{split}' split...")
        
        split_input_path = input_data_path / split
        split_output_path = output_data_path / split
        
        audio_files = list(split_input_path.glob("**/*.wav")) + list(split_input_path.glob("**/*.mp3"))
        
        for audio_file in tqdm(audio_files, desc=f"Reading {split} files"):
            spectrogram = process_audio_file(audio_file)
            if spectrogram is not None:
                
                class_name = audio_file.parent.name
                
                output_class_dir = split_output_path / class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                final_path = output_class_dir / f"{audio_file.stem}.npy"

                all_spectrograms[split].append((spectrogram, final_path))

    print("\nCalculating normalization statistics from the training set...")
    train_spectrograms_list = [s[0] for s in all_spectrograms['train']]
    if not train_spectrograms_list:
        print("Error: No training files were processed. Cannot calculate normalization stats.")
        return
        
    train_spectrograms_arr = np.stack(train_spectrograms_list)
    
    mean = np.mean(train_spectrograms_arr, axis=(0, 2), keepdims=True)
    std = np.std(train_spectrograms_arr, axis=(0, 2), keepdims=True)
    std[std == 0] = 1e-8
    
    print("Normalization stats calculated. Applying normalization and saving files...")

    for split in ['train', 'val', 'test']:
        for spectrogram, final_path in tqdm(all_spectrograms[split], desc=f"Saving {split} files"):
            normalized_spectrogram = (spectrogram - mean) / std
            np.save(final_path, normalized_spectrogram.astype(np.float32))

    print("\nPreprocessing complete!")
    print(f"Processed files are saved in: {output_data_path}")

if __name__ == "__main__":
    main()