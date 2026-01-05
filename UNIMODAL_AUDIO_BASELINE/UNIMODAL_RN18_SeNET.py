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
from sklearn.metrics import f1_score
import math
import warnings
warnings.filterwarnings("ignore") 

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Set the desired GPU device


# --- CONFIGURATION ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WORKERS = 16 
DEFAULT_WORKERS = min(MAX_WORKERS, os.cpu_count() // 2) if os.cpu_count() else 4
PIN_MEMORY = True if DEVICE == "cuda" else False


BASE_DIR = Path("../ADVANCE_features") 
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "test"
NORMALIZER_PATH = BASE_DIR / "normalizer_train.npy"
# Checkpoint name changed to reflect L4-only SEBlock
CHECKPOINT_PATH = "best_rn18_advance_audio_senet_l4only.pth" 

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


# --- DATA TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
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
        sound = (sound - self.mu) / self.sigma
        sound = torch.from_numpy(sound).float()
        if sound.ndim > 2:
            sound = sound.squeeze()
        sound = sound.unsqueeze(0)
        # Convert to 3-channel for pre-trained ResNet
        sound = sound.expand(3, -1, -1)
        if self.transform:
            sound = self.transform(sound)
        return sound, label


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=16, time_mask_param=50, num_freq_masks=1, num_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def forward(self, x):
        if not self.training:
            return x
        
        clone = x.clone()
        B, C, H, W = clone.shape
        
        # Frequency Masking (dim H)
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_param))
            if H - f > 0 and f > 0:
                f0 = int(np.random.uniform(0, H - f))
                clone[:, :, f0:f0 + f, :] = 0

        # Time Masking (dim W)
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_param))
            if W - t > 0 and t > 0:
                t0 = int(np.random.uniform(0, W - t))
                clone[:, :, :, t0:t0 + t] = 0
        return clone


# ==================== SENet Components (Unchanged) ====================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention only)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Squeeze operation: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation: Two fully-connected layers (MLP)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        
        # 1. Squeeze: Global Average Pooling -> [B, C, 1, 1]
        y = self.avg_pool(x).view(B, C)
        
        # 2. Excitation: MLP -> [B, C]
        y = self.excitation(y)
        
        # Reshape to [B, C, 1, 1] for broadcasting and scale
        y = y.view(B, C, 1, 1)
        
        # 3. Scale: Element-wise multiplication
        return x * y.expand_as(x)


# ==================== ResNet18 with SENet (Layer 4 Only) ====================

class ResNet18_SENet_L4Only(nn.Module):
    def __init__(self, num_classes, reduction=16):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Copy standard ResNet layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Instantiate **ONLY** SEBlock after stage 4
        # Channel count for layer4 is 512
        self.se4 = SEBlock(512, reduction=reduction)
        
        self.avgpool = base_model.avgpool
        
        # Reconstruct FC layers
        num_ftrs = base_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(num_ftrs // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x) # No SEBlock here
        
        x = self.layer3(x) # No SEBlock here
        
        x = self.layer4(x)
        x = self.se4(x) # <--- SENet only after layer4 (C=512)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    train_ds = OfflineAudioDataset(root_dir=TRAIN_DIR, mu=mu, sigma=sigma, transform=train_transform)
    val_ds = OfflineAudioDataset(root_dir=VAL_DIR, mu=mu, sigma=sigma, transform=val_transform)


    class_names = list(train_ds.class_to_idx.keys())
    NUM_CLASSES = len(class_names)
    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CLASSES} classes: {', '.join(class_names)}")
    print(f"Train samples: {len(train_ds)} | Test samples: {len(val_ds)}\n")

    # --- Data Loading and Balancing Setup ---
    class_names_list = [class_name for _, class_name in train_ds.samples]
    class_counts = Counter(class_names_list)
    
    try:
        sample_weights = [1.0 / class_counts[class_name] for _, class_name in train_ds.samples]
    except KeyError as e:
        print(f"Error: Missing class count for class {e}. Check dataset integrity.")
        raise
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
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


    # Model with SENet (Layer 4 Only)
    model = ResNet18_SENet_L4Only(NUM_CLASSES, reduction=16).to(DEVICE)
    spec_augmenter = SpecAugment().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1 = 0.0
    patience_counter = 0


    for epoch in range(1, EPOCHS + 1):
        model.train()
        spec_augmenter.train() 
        running_loss, running_correct, running_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Apply SpecAugment during training
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
        
        spec_augmenter.eval() 
        val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, DEVICE, set_name="Test") 
        scheduler.step()


        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved improved checkpoint (test_f1={val_f1:.4f}) to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")


        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break


    print("\n--- Training Complete ---")
    if Path(CHECKPOINT_PATH).exists():
        print("\n--- Evaluating Best Model on TEST Set ---")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval() 
        test_loss, test_acc, test_f1 = evaluate(val_loader, model, criterion, DEVICE, set_name="Test")
        print(f"\nFinal Metrics on TEST Set:\nLoss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1-Score: {test_f1:.4f}")
    else:
        print("No checkpoint found to evaluate.")


    print("\n✅ Done.")