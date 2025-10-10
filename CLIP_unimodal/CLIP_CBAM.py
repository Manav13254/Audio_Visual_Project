import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import clip
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, classification_report
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"
BASE_DIR = Path(r"E:\Audio_Visual_Project\final_data_split_80_20")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
CHECKPOINT_PATH = "best_image_clip_cbam_checkpoint.pt"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

clip_model, preprocess = clip.load(BACKBONE, device=DEVICE, jit=False)
clip_model.eval()

IMAGE_SIZE = clip_model.visual.input_resolution
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
val_transform = preprocess

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.vision_root = Path(root_dir) / "vision"
        self.classes = sorted([d.name for d in self.vision_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_name in self.classes:
            class_vision_dir = self.vision_root / class_name
            for image_path in class_vision_dir.glob('*.jpg'):
                label = self.class_to_idx[class_name]
                samples.append((str(image_path), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

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
        return self.sigmoid(avg_out + max_out)

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

class CLIP_RN50_CBAM_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        self.cbam2 = CBAM(self.clip_visual.layer2[-1].conv3.out_channels)
        self.cbam3 = CBAM(self.clip_visual.layer3[-1].conv3.out_channels)
        self.cbam4 = CBAM(self.clip_visual.layer4[-1].conv3.out_channels)
        
        output_dim = self.clip_visual.output_dim
        hidden_dim = output_dim // 2
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        v = self.clip_visual
        x = images.type(v.conv1.weight.dtype)
        
        x = v.conv1(x); x = v.bn1(x); x = v.relu(x); x = v.maxpool(x)
        x = v.layer1(x)
        x = v.layer2(x); x = self.cbam2(x)
        x = v.layer3(x); x = self.cbam3(x)
        x = v.layer4(x); x = self.cbam4(x)
        x = v.attnpool(x)
        
        return self.head(x.float())

@torch.no_grad()
def evaluate(net, loader, criterion, class_names, print_report=False):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            out = net(imgs)
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

if __name__ == '__main__':
    train_ds = ImageFolderDataset(TRAIN_DIR, transform=train_transform)
    val_ds = ImageFolderDataset(VAL_DIR, transform=val_transform)
    
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [item[1] for item in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CLIP_RN50_CBAM_Head(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = list(model.head.parameters()) + list(model.cbam2.parameters()) + \
                       list(model.cbam3.parameters()) + list(model.cbam4.parameters())
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    mlflow.set_experiment("image_clip_cbam_tuning")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE):
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                running_corrects += (out.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_ds)
            train_acc = 100.0 * running_corrects / len(train_ds)
            
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, train_ds.classes)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}% | Val F1={val_f1:.4f}")

            if val_f1 > best_val_f1 + MIN_DELTA:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"   -> Saved improved checkpoint (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"   -> No improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\nLoading best model for final evaluation.")
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            final_loss, final_acc, final_f1 = evaluate(model, val_loader, criterion, train_ds.classes, print_report=True)
            mlflow.log_metrics({
                "final_val_loss": final_loss, "final_val_acc": final_acc, "final_val_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "image_clip_cbam_model")

    print("\n✅ Done.")
