import os
import random
import numpy as np
import math
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
import warnings
warnings.filterwarnings("ignore") # Suppress unnecessary warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the desired GPU device


# --- CONFIGURATION (Modified for performance) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Performance settings
MAX_WORKERS = 16 
DEFAULT_WORKERS = min(MAX_WORKERS, os.cpu_count() // 2) if os.cpu_count() else 4
PIN_MEMORY = True if DEVICE == "cuda" else False


BASE_DIR = Path("../ADVANCE_features") 
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "test" 
NORMALIZER_PATH = BASE_DIR / "normalizer_train.npy"
CHECKPOINT_PATH = "best_rn18_advance_audio.pth"

BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 15
MIN_DELTA = 5e-4

# --- LOAD NORMALIZER ---
if not NORMALIZER_PATH.exists():
    raise FileNotFoundError(f"Normalizer file not found: {NORMALIZER_PATH}")
normalizer = np.load(NORMALIZER_PATH)
mu, sigma = normalizer[0], normalizer[1]

# --- DATA TRANSFORMS (Removed RandomErasing as it's redundant with SpecAugment) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    # transforms.RandomErasing removed here
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
])

# --- DATASET (Unchanged) ---
class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, mu, sigma, transform=None):
        self.transform = transform
        self.mu = mu
        self.sigma = sigma
        self.samples = self._find_samples(root_dir)
        self.class_to_idx = self._get_class_to_idx(root_dir)
    def _find_samples(self, root_dir):
        samples = []
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Data root directory not found: {root}")
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
        for class_name in class_names:
            class_dir = root / class_name
            for feature_path in class_dir.glob('*.npy'):
                samples.append((feature_path, class_name))
        return samples
    def _get_class_to_idx(self, root_dir):
        root = Path(root_dir)
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
        return {name: i for i, name in enumerate(class_names)}
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        feature_path, class_name = self.samples[index]
        label = self.class_to_idx[class_name]
        sound = np.load(feature_path)
        # Normalize using train-set mu and sigma
        sound = (sound - self.mu) / self.sigma
        sound = torch.from_numpy(sound).float()
        # Add single channel and expand to 3 channels for pretrained ResNet
        if sound.ndim > 2:
            sound = sound.squeeze()
        sound = sound.unsqueeze(0)
        sound = sound.expand(3, -1, -1)
        if self.transform:
            sound = self.transform(sound)
        return sound, label

# --- SpecAugment (Unchanged) ---
class SpecAugment(nn.Module):
    # Simple SpecAugment time/frequency masking on spectrograms [3,H,W]
    def __init__(self, freq_mask_param=16, time_mask_param=50, num_freq_masks=1, num_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    def forward(self, x):
        clone = x.clone()
        # Frequency masking (on H dimension)
        _, _, H, W = clone.shape
        for _ in range(self.num_freq_masks):
            # Masking should be applied to the frequency dimension (H)
            f = int(np.random.uniform(0, self.freq_mask_param))
            if H - f > 0:
                f0 = int(np.random.uniform(0, H - f))
                clone[:, :, f0:f0 + f, :] = 0
        # Time masking (on W dimension)
        for _ in range(self.num_time_masks):
            # Masking should be applied to the time dimension (W)
            t = int(np.random.uniform(0, self.time_mask_param))
            if W - t > 0:
                t0 = int(np.random.uniform(0, W - t))
                clone[:, :, :, t0:t0 + t] = 0
        return clone

@torch.no_grad()
def evaluate(data_loader, net, criterion, device, set_name="Validation"):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
            out = net(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = loss_sum / total
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1

if __name__ == '__main__':
    train_ds = OfflineAudioDataset(root_dir=TRAIN_DIR, mu=mu, sigma=sigma, transform=train_transform)
    val_ds = OfflineAudioDataset(root_dir=VAL_DIR, mu=mu, sigma=sigma, transform=val_transform)

    class_names = list(train_ds.class_to_idx.keys())
    NUM_CLASSES = len(class_names)
    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CLASSES} classes: {', '.join(class_names)}")
    print(f"Train samples: {len(train_ds)} | Test samples: {len(val_ds)}\n")

    labels = [lbl for _, lbl in train_ds.samples]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[l] for _, l in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    # 🚀 Applying performance improvements
    print(f"Using {DEFAULT_WORKERS} workers with pin_memory={PIN_MEMORY}.")
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=DEFAULT_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(num_ftrs // 2, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    spec_augmenter = SpecAugment().to(DEVICE) # SpecAugment remains a separate GPU module
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # 🔥 Apply SpecAugment on the GPU batch
            augmented_imgs = spec_augmenter(imgs)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
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
        val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, DEVICE, set_name="Test")
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"   -> Saved improved checkpoint (test_f1={val_f1:.4f}) to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"   -> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\n--- Training Complete ---")
    
    # Final evaluation on the test set
    if Path(CHECKPOINT_PATH).exists():
        print("\n--- Evaluating Best Model on TEST Set ---")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        
        # Collect final metrics and predictions
        model.eval()
        all_labels = []
        final_preds = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
                    out = model(imgs)
                preds = out.argmax(dim=1).cpu().numpy()
                final_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        test_loss, test_acc, test_f1 = evaluate(val_loader, model, criterion, DEVICE, set_name="Test")
        print(f"\nFinal Metrics on TEST Set:\nLoss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1-Score: {test_f1:.4f}")
        
        target_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
        if class_names:
            target_names = class_names

        print("\nClassification Report:")
        print(classification_report(all_labels, final_preds, target_names=target_names, zero_division=0))
    else:
        print("No checkpoint found to evaluate.")

    print("\n✅ Done.")