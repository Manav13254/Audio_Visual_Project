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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torch.nn.functional as F

import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# --- Set PyTorch to use GPU 1 (if you have multiple GPUs) ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# ---------------- Config / Seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- PATHS UPDATED for the clean SPECTROGRAM data ---
DATA_DIR = "../final_data" 
TRAIN_DIR = os.path.join(DATA_DIR, "train", "spectrograms")
VAL_DIR = os.path.join(DATA_DIR, "val", "spectrograms")
# ---

BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_DELTA = 5e-4

# --- CHECKPOINT PATH UPDATED for the clean run ---
CHECKPOINT_PATH = "best_finetuned_rn18_bam_full_backbone_CLEAN_checkpoint.pt"
# ---

# --- Use standard ImageNet normalization ---
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# ---------------- Transforms for Spectrogram Images ----------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.10), ratio=(0.2, 0.8), value='random'),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.12), ratio=(3.0, 8.0), value='random'),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

# ---------------- Datasets & Sampler ----------------
print(f"Loading data from {os.path.abspath(TRAIN_DIR)} and {os.path.abspath(VAL_DIR)}")
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

labels = [label for _, label in train_ds.samples]
class_counts = Counter(labels)
num_samples = len(train_ds)
class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

NUM_CLASSES = len(train_ds.classes)
print(f"Found {NUM_CLASSES} classes: {', '.join(train_ds.classes)}")

# ---------------- BAM Module Definition ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc(avg_out.view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        return avg_out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv1(max_out)

class BAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        
        attention_map = torch.sigmoid(channel_out + spatial_out)
        return x * attention_map

# ---------------- Model (Full Fine-tuning of ResNet-18 with BAM) ----------------
class ResNet18_BAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_BAM, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.bam0 = BAM(in_planes=128) # After layer2
        self.bam1 = BAM(in_planes=256) # After layer3
        self.bam2 = BAM(in_planes=512) # After layer4
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(num_ftrs // 2, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.bam0(x)
        
        x = self.backbone.layer3(x)
        x = self.bam1(x)
        
        x = self.backbone.layer4(x)
        x = self.bam2(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x

# ---------------- Model Instantiation ----------------
print("\nLoading pre-trained ResNet-18 with BAM...")
model = ResNet18_BAM(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

print(f"Fine-tuning ResNet-18 with BAM on {DEVICE}...")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable parameters (%): {100.0 * trainable_params / total_params:.2f}%")

# ---------------- Optimizer / Loss / Scheduler ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------- Evaluate Function ----------------
@torch.no_grad()
def evaluate(net, print_report=False):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            out = net(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = loss_sum / total
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    if print_report:
        print(f"\nVal Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% | Val F1: {f1:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
        print(classification_report(all_labels, all_preds, target_names=val_ds.classes, zero_division=0))
    return avg_loss, acc, f1

# ---------------- Main Execution Block ----------------
def main():
    mlruns_path = Path.cwd() / "mlruns"
    mlruns_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment("sound_classification_rn18_bam_CLEAN")
    
    with mlflow.start_run():
        mlflow.log_params({
            "data_split": "clean_grouped_split",
            "epochs": EPOCHS, 
            "lr": LR,
            "batch_size": BATCH_SIZE, 
            "architecture": "ResNet-18 + BAM Full Fine-tuned", 
            "regularization": f"Dropout (p=0.7), WeightDecay ({WEIGHT_DECAY})"
        })
        
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
            
            current_lr = optimizer.param_groups[0]['lr']
            
            val_loss, val_acc, val_f1 = evaluate(model)
            scheduler.step()

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% | LR={current_lr:.6f}")
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc, 
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": current_lr
            }, step=epoch)

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> Saved improved checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        final_loss, final_acc, final_f1 = evaluate(model, print_report=True)
        mlflow.log_metrics({"final_val_loss": final_loss, "final_val_acc": final_acc, "final_val_f1": final_f1})
        mlflow.pytorch.log_model(model, "sound_classifier_rn18_bam_clean_model")

    print("\nDone.")

if __name__ == '__main__':
    main()