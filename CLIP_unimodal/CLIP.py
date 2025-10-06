# train_audiovisual_clip.py

import os
import random
import numpy as np
import json
from collections import Counter
from tqdm import tqdm

# --- GPU Configuration ---
# Set the GPU you want to use. "0", "1", etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets # For the zero-shot baseline
import clip
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, log_loss
from PIL import Image
import torchaudio

# ---------------- Config / Seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"
JSON_FILE_PATH = os.path.join('DATASET', 'dataset_split.json')
CHECKPOINT_PATH = "best_audiovisual_clip_checkpoint.pt"

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

# --- Audio/Visual Config ---
AUDIO_DURATION_SECONDS = 10
TARGET_TIME_FRAMES = 256
SAMPLE_RATE = 16000

# ---------------- Load CLIP Model and Preprocessor ----------------
# Load to CPU first to avoid taking up GPU memory if we only need the preprocessor
clip_model, preprocess = clip.load(BACKBONE, device="cpu")
# The validation transform for images is the standard CLIP preprocessor
val_image_transform = preprocess

# ---------------- Custom AudioVisualDataset ----------------
class AudioVisualDataset(Dataset):
    def __init__(self, data_dict, image_transform, audio_transform, target_sample_rate, audio_len_seconds, target_time_frames):
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.target_sample_rate = target_sample_rate
        self.num_audio_samples = self.target_sample_rate * audio_len_seconds
        self.target_time_frames = target_time_frames
        
        self.file_list = []
        for key, value in data_dict.items():
            self.file_list.append((value['image'], value['audio'], value['label']))

    def __len__(self):
        return len(self.file_list)

    def _process_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] > self.num_audio_samples:
            waveform = waveform[:, :self.num_audio_samples]
        else:
            num_missing = self.num_audio_samples - waveform.shape[1]
            padding = (0, num_missing)
            waveform = torch.nn.functional.pad(waveform, padding)
        return waveform

    def __getitem__(self, idx):
        image_path, audio_path, label = self.file_list[idx]
        
        image = Image.open(image_path).convert('RGB')
        waveform = self._process_audio(audio_path)

        image = self.image_transform(image)
        spectrogram = self.audio_transform(waveform)

        current_time_frames = spectrogram.shape[2]
        if current_time_frames > self.target_time_frames:
            spectrogram = spectrogram[:, :, :self.target_time_frames]
        elif current_time_frames < self.target_time_frames:
            padding_needed = self.target_time_frames - current_time_frames
            spectrogram = F.pad(spectrogram, (0, padding_needed))

        return image, spectrogram, label

# ---------------- New Audio-Visual Model ----------------
class CLIPAudioModel(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
            
        for param in self.clip_visual.parameters():
            param.requires_grad = False
            
        num_vision_features = self.clip_visual.output_dim
        
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        num_audio_features = 64
        
        hidden_dim = (num_vision_features + num_audio_features) // 2
        self.head = nn.Sequential(
            nn.Linear(num_vision_features + num_audio_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, audio_spectrograms):
        image_features = self.clip_visual(images.type(self.clip_visual.conv1.weight.dtype))
        audio_features = self.audio_backbone(audio_spectrograms)
        audio_features = audio_features.view(audio_features.size(0), -1)
        combined_features = torch.cat((image_features, audio_features), dim=1)
        return self.head(combined_features.float())

# ---------------- Evaluation Functions ----------------
@torch.no_grad()
def evaluate(net, loader, criterion, class_names, print_report=False):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels = [], []
    for imgs, audios, labels in loader:
        imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
        
        with torch.amp.autocast(device_type=DEVICE):
            out = net(imgs, audios)
            loss = criterion(out, labels)
            
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = loss_sum / len(all_labels)
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    if print_report:
        print(f"\n--- Final Validation Report ---")
        print(f"Val loss: {avg_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    return avg_loss, acc, f1

# ---------------- Main Script Execution ----------------
if __name__ == '__main__':
    # 1. Load Data Splits from JSON
    with open(JSON_FILE_PATH, 'r') as f:
        data_splits = json.load(f)
    train_data_dict = data_splits['train']
    val_data_dict = data_splits['val']
    class_to_idx = data_splits['class_to_idx']
    class_names = list(class_to_idx.keys())
    NUM_CLASSES = len(class_to_idx)
    
    print(f"Found {NUM_CLASSES} classes: {class_names}")

    # 2. Define Transforms
    audio_transforms = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
    )

    # 3. Create Datasets
    train_ds = AudioVisualDataset(train_data_dict, val_image_transform, audio_transforms, SAMPLE_RATE, AUDIO_DURATION_SECONDS, TARGET_TIME_FRAMES)
    val_ds = AudioVisualDataset(val_data_dict, val_image_transform, audio_transforms, SAMPLE_RATE, AUDIO_DURATION_SECONDS, TARGET_TIME_FRAMES)

    # 4. Create WeightedRandomSampler
    train_labels = [item[2] for item in train_ds.file_list]
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 6. Instantiate Model, Loss, Optimizer, etc.
    clip_model.to(DEVICE) # Move the original CLIP model to the GPU now
    model = CLIPAudioModel(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 7. MLflow Setup
    mlflow.set_experiment("audiovisual_clip_fusion_training")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

        best_val_loss = float("inf")
        patience_counter = 0

        # 8. Training Loop
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            for imgs, audios, labels in pbar:
                imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=DEVICE):
                    out = model(imgs, audios)
                    loss = criterion(out, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * imgs.size(0)
                running_corrects += (out.argmax(dim=1) == labels).sum().item()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_corrects/len(train_ds):.2f}%")

            train_loss = running_loss / len(train_ds)
            train_acc = 100.0 * running_corrects / len(train_ds)
            
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, class_names)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> Saved improved checkpoint to {CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
        # 9. Final Evaluation
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation.")
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            
            final_loss, final_acc, final_f1 = evaluate(model, val_loader, criterion, class_names, print_report=True)
            mlflow.log_metrics({
                "final_val_loss": final_loss, "final_val_acc": final_acc, "final_val_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "audiovisual_clip_model")

    print("\nDone.")