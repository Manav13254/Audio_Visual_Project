import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# --- CONFIGURATION ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path("/home/23ucs671/audio_visual_proj1/preprocessed_features_80_20")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
CHECKPOINT_PATH = "best_finetuned_rn18_model_(offline).pt"

normalizer = np.load(BASE_DIR / "normalizer.npy")
mu, sigma = normalizer[0], normalizer[1]

# ----------------- Hyperparameters -----------------
BATCH_SIZE = 32
EPOCHS = 100
LR = 4e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 15
MIN_DELTA = 5e-4

# ----------------- Data Transforms (for Tensors) -----------------
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.class_to_idx = {}
        self.samples = self._find_samples(root_dir)

    def _find_samples(self, root_dir):
        samples = []
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        for class_name in class_names:
            class_dir = root / class_name
            for feature_path in class_dir.rglob('*.npy'):
                samples.append((feature_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        feature_path, label = self.samples[index]
        
        sound = np.load(feature_path)
        sound = (sound - mu) / sigma
        sound = torch.from_numpy(sound).unsqueeze(0)
        sound = sound.expand(3, -1, -1)
        
        if self.transform:
            sound = self.transform(sound)
            
        return sound, label

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=40, time_mask_param=70, num_freq_masks=1, num_time_masks=1):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    def forward(self, x):
        for _ in range(self.num_freq_masks):
            f = np.random.uniform(0, self.freq_mask_param); f0 = np.random.uniform(0, x.shape[2] - f); x[:, :, int(f0):int(f0 + f), :] = 0
        for _ in range(self.num_time_masks):
            t = np.random.uniform(0, self.time_mask_param); t0 = np.random.uniform(0, x.shape[3] - t); x[:, :, :, int(t0):int(t0 + t)] = 0
        return x

@torch.no_grad()
def evaluate(data_loader, net, criterion, device, set_name="Validation"):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device):
            out = net(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = loss_sum / total
    acc = 100.0 * correct / total
    return avg_loss, acc

if __name__ == '__main__':
    train_ds = OfflineAudioDataset(root_dir=TRAIN_DIR, transform=train_transform)
    val_ds = OfflineAudioDataset(root_dir=VAL_DIR, transform=val_transform)
    
    labels = [label for _, label in train_ds.samples]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = list(train_ds.class_to_idx.keys())
    NUM_CLASSES = len(class_names)
    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CLASSES} classes: {', '.join(class_names)}")
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}\n")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters(): param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(), nn.Dropout(p=0.7), nn.Linear(num_ftrs // 2, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    
    spec_augmenter = SpecAugment().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # --- Variables to track best metrics ---
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            augmented_imgs = spec_augmenter(imgs)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                out = model(augmented_imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_samples += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_correct/running_samples:.2f}%")
            
        train_loss = running_loss / running_samples
        train_acc = 100.0 * running_correct / running_samples
        
        val_loss, val_acc = evaluate(val_loader, model, criterion, DEVICE, set_name="Validation")
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            # --- Store the best metrics from this epoch ---
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"    -> Saved improved checkpoint (val_loss={val_loss:.4f}) to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"    -> No improvement. Patience {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print("\n--- Training Complete ---")
    if Path(CHECKPOINT_PATH).exists():
        print(f"Best model saved to {CHECKPOINT_PATH} based on validation loss.")
        
        # --- Print the best recorded metrics from training ---
        print("\n--- Best Metrics (from epoch with lowest validation loss) ---")
        print(f"Train Loss: {best_train_loss:.4f}")
        print(f"Train Acc:  {best_train_acc:.2f}%")
        print(f"Val Loss:   {best_val_loss:.4f}")
        print(f"Val Acc:    {best_val_acc:.2f}%")
        
        # --- Evaluate the best model and calculate F1 score ---
        print("\n--- Final Evaluation on Validation Set (using best model) ---")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE):
                    out = model(imgs)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # --- Calculate and print F1 score and classification report ---
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Weighted F1 Score: {f1:.4f}\n")
        
        print("Classification Report:")
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print(report)

    else:
        print("Training finished, but no checkpoint was saved.")
    
    print("\n✅ Done.")