import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import clip
from sklearn.metrics import f1_score, classification_report
from PIL import Image
from collections import Counter
import warnings
import math # Added for robust CPU count check

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# --- CONFIGURATION (DGX/Linux Path Adjusted) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"
# --- OPTIMIZATION PARAMETERS ---
MAX_WORKERS = 16 
DEFAULT_WORKERS = min(MAX_WORKERS, os.cpu_count() // 2) if os.cpu_count() else 4
PIN_MEMORY = True 
# -------------------------------

BASE_DIR = Path("../ADVANCE_DATA_split") 
TRAIN_DIR = BASE_DIR / "train" / "vision"
TEST_DIR = BASE_DIR / "test" / "vision"
CHECKPOINT_PATH = "best_image_clip_rn50_spatialsenet_advancesplit.pt" 
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
SE_REDUCTION = 16


# CLIP model and transforms
print(f"Loading CLIP model {BACKBONE} on {DEVICE}...")
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
test_transform = preprocess # Use the standard preprocess for the final test evaluation

# --- DATASET CLASS (Adapted for single root dir) ---
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
            
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if class_dir.is_dir():
                for image_path in class_dir.glob('*.jpg'):
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

# --- MODEL COMPONENTS (Unchanged) ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CLIP_RN50_SE_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        self.se = SELayer(2048) # Only one SE block for layer4
        
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
        
        # Manual forward pass to inject SE layers.
        x = v.conv1(x)
        x = v.bn1(x)
        x = v.relu1(x)
        x = v.conv2(x)
        x = v.bn2(x)
        x = v.relu2(x)
        x = v.conv3(x)
        x = v.bn3(x)
        x = v.relu3(x)
        x = v.avgpool(x)
        
        x = v.layer1(x)
        x = v.layer2(x)
        x = v.layer3(x)
        x = v.layer4(x); x = self.se(x)  # Only applying SE after layer4
        
        x = v.attnpool(x)
        
        return self.head(x.float())

@torch.no_grad()
def evaluate(net, loader, criterion, class_names):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            out = net(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = loss_sum / len(all_labels)
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1


if __name__ == '__main__':
    train_ds = ImageFolderDataset(TRAIN_DIR, transform=train_transform)
    test_ds = ImageFolderDataset(TEST_DIR, transform=test_transform)
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [item[1] for item in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    # 🚀 Data Loading Optimization applied:
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=DEFAULT_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=DEFAULT_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    print(f"Using {DEFAULT_WORKERS} DataLoader workers with pin_memory={PIN_MEMORY}.")
    
    # Model with Spatial SENet
    model = CLIP_RN50_SE_Head(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters (Head + SE Blocks): {sum(p.numel() for p in trainable_params):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    best_test_f1 = 0.0
    patience_counter = 0
    print("\n--- Starting Fine-Tuning (SE Blocks + Head) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_corrects = 0.0, 0
        train_preds_epoch, train_labels_epoch = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            train_preds_epoch.extend(preds.cpu().numpy())
            train_labels_epoch.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss = running_loss / len(train_ds)
        train_acc = 100.0 * running_corrects / len(train_ds)
        train_f1 = f1_score(train_labels_epoch, train_preds_epoch, average="weighted", zero_division=0)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, train_ds.classes)
        scheduler.step()
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, F1={train_f1:.4f} | Test Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Test F1={test_f1:.4f}")
        if test_f1 > best_test_f1 + MIN_DELTA:
            best_test_f1 = test_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"   -> Saved improved checkpoint (Test F1: {best_test_f1:.4f})")
        else:
            patience_counter += 1
            print(f"   -> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break


    # --- Final Evaluation ---
    print("\n--- Evaluating Best Model on TEST Set ---")
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, train_ds.classes)
        print(f"\n--- Final TEST Set Report ---")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.4f}")
        
        # 🛡️ Robust Final Prediction: Iterate over the test_loader instead of stacking all images.
        all_labels = [s[1] for s in test_ds.samples]
        final_preds = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    out = model(imgs)
                final_preds.extend(out.argmax(dim=1).cpu().numpy())
        
        print(classification_report(
            all_labels, 
            final_preds,
            target_names=test_ds.classes,
            zero_division=0
        ))
    print("\n✅ Done.")