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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(r"/home/23ucs671/audio_visual_proj1/preprocessed_features_80_20")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
CHECKPOINT_PATH = "best_rn18_cbam_model_80_20.pt"

try:
    normalizer = np.load(BASE_DIR / "normalizer.npy")
    mu, sigma = normalizer[0], normalizer[1]
except FileNotFoundError:
    print("Error: normalizer.npy not found. Please run the feature extraction script first.")
    exit()

BATCH_SIZE = 32
EPOCHS = 100
LR = 4e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 15
MIN_DELTA = 5e-4

train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = self._find_samples(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(root_dir)))}

    def _find_samples(self, root_dir):
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        return sorted(list(root.rglob('*.npy')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        feature_path = self.samples[index]
        label = self.class_to_idx[feature_path.parent.name]
        sound = np.load(feature_path)
        sound = (sound - mu) / sigma
        sound = torch.from_numpy(sound).unsqueeze(0)
        sound = sound.expand(3, -1, -1)
        if self.transform:
            sound = self.transform(sound)
        return sound, label

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

@torch.no_grad()
def evaluate(data_loader, net, criterion, device):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device):
            out = net(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total

if __name__ == '__main__':
    train_ds = OfflineAudioDataset(root_dir=TRAIN_DIR, transform=train_transform)
    val_ds = OfflineAudioDataset(root_dir=VAL_DIR, transform=val_transform)
    
    labels = [ds.class_to_idx[path.parent.name] for path in train_ds.samples]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Using device: {DEVICE}")
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}\n")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    model.layer2.add_module("CBAM", CBAM(128))
    model.layer3.add_module("CBAM", CBAM(256))
    model.layer4.add_module("CBAM", CBAM(512))
    
    for param in model.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(), nn.Dropout(p=0.7),
        nn.Linear(num_ftrs // 2, len(train_ds.class_to_idx))
    )
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (out.argmax(dim=1) == labels).sum().item()
            running_samples += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_correct/running_samples:.2f}%")
        
        train_loss = running_loss / running_samples
        train_acc = 100.0 * running_correct / running_samples
        val_loss, val_acc = evaluate(val_loader, model, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"   -> Saved improved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   -> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
            
    print("\n✅ Training complete!")
