# Filename: train_bam_resnet512.py

import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report
from typing import Tuple

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path("../data/processed_audio")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
CHECKPOINT_PATH = "bam_resnet18_512.pth"

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
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
])

class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.class_to_idx, self.samples = self._make_dataset(root_dir)
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

    def _make_dataset(self, root_dir: Path) -> Tuple[dict, list]:
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        samples = []
        for class_name in class_names:
            class_idx = class_to_idx[class_name]
            for feature_path in (root / class_name).rglob("*.npy"):
                samples.append((feature_path, class_idx))
        return class_to_idx, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        feature_path, label = self.samples[index]
        sound = np.load(feature_path)
        sound_tensor = torch.from_numpy(sound).float()
        if sound_tensor.ndim > 2:
            sound_tensor = sound_tensor.squeeze()
        sound_tensor = sound_tensor.unsqueeze(0).expand(3, -1, -1)  
        if self.transform:
            sound_tensor = self.transform(sound_tensor)
        return sound_tensor, label


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
        att = self.mlp(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)
        return att

class SpatialGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.conv1x1 = nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1)
        self.conv_list = nn.ModuleList([
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels // reduction_ratio,
                      kernel_size=3, padding=dilation_val, dilation=dilation_val)
            for _ in range(dilation_conv_num)
        ])
        self.conv_out = nn.Conv2d(gate_channels // reduction_ratio, 1, kernel_size=1)
    def forward(self, x):
        x = self.conv1x1(x)
        for conv in self.conv_list:
            x = F.relu(conv(x))
        return self.conv_out(x)

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self, x):
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x


@torch.no_grad()
def evaluate(loader, model, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(imgs)
            loss = criterion(out, labels)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = loss_sum / total
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1


if __name__ == "__main__":
    train_ds = OfflineAudioDataset(TRAIN_DIR, train_transform)
    val_ds = OfflineAudioDataset(VAL_DIR, val_transform)

    class_counts = Counter([lbl for _, lbl in train_ds.samples])
    weights = [1.0 / class_counts[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    NUM_CLASSES = len(train_ds.class_to_idx)
    print(f"Using {DEVICE} | Classes: {NUM_CLASSES}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.layer2.add_module("BAM", BAM(128))
    model.layer3.add_module("BAM", BAM(256))
    model.layer4.add_module("BAM", BAM(512))

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_f1 = 0.0
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        all_train_preds, all_train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

        val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.2f}% | TrainF1={train_f1:.4f} | "
              f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.2f}% | ValF1={val_f1:.4f}")

        if val_f1 > best_f1 + MIN_DELTA:
            best_f1, patience = val_f1, 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"✅ Saved best model (Val F1={val_f1:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("⏹️ Early stopping")
                break

    print("Training complete!")

    if TEST_DIR.exists():
        test_ds = OfflineAudioDataset(TEST_DIR, val_transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()

        test_loss, test_acc, test_f1 = evaluate(test_loader, model, criterion, DEVICE)
        print(f"\nTestLoss={test_loss:.4f} | TestAcc={test_acc:.2f}% | TestF1={test_f1:.4f}")

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        class_names = [test_ds.idx_to_class[i] for i in range(len(test_ds.class_to_idx))]
        print("\nClassification Report (per-class metrics):")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
