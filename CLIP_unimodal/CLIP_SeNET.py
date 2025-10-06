import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # You can use "1", "2", "3", or "5"


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import clip
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss

# ---------------- Config / Seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"
TRAIN_DIR = "../final_data/train/vision"
VAL_DIR = "../final_data/val/vision"
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-5 # Lower LR as the model is more complex
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
CHECKPOINT_PATH = "best_clip_senet_head_checkpoint.pt" # <-- CHANGED
IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ---------------- Load CLIP backbone ----------------
clip_model, preprocess = clip.load(BACKBONE, device=DEVICE, jit=False)
val_transform = preprocess
clip_model.to(DEVICE).float()
clip_model.eval()

# ---------------- Transforms ----------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

# ---------------- Datasets & Sampler ----------------
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

labels = [label for _, label in train_ds.imgs]
class_counts = Counter(labels)
num_samples = len(train_ds)
class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_ds.classes)


# ---------------- Squeeze-and-Excitation (SE) Block Implementation ----------------
class SEBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------- CLIP Model with SENet and Classification Head ----------------
class CLIP_RN50_SENet(nn.Module):
    def __init__(self, clip_model, n_cls, dropout_rate=0.5):
        super().__init__()
        self.clip = clip_model
        
        # Freeze the entire CLIP model
        for param in self.clip.parameters():
            param.requires_grad = False
            
        v = self.clip.visual
        
        # --- Dynamically create SE modules for layers 2, 3, 4 ---
        self.se_modules = nn.ModuleList()
        # Create a dummy tensor to find the output channel dimensions
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE, dtype=v.conv1.weight.dtype)

        with torch.no_grad():
            # FIX: Replaced incorrect stem (with v.maxpool) with the correct ModifiedResNet stem from CLIP
            x = v.conv1(dummy); x = v.bn1(x); x = v.relu1(x)
            x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
            x = v.conv3(x); x = v.bn3(x); x = v.relu3(x)
            x = v.avgpool(x)
            # --- END FIX ---
            x = v.layer1(x)
            x = v.layer2(x)
            self.se_modules.append(SEBlock(x.shape[1])) # After Layer 2
            x = v.layer3(x)
            self.se_modules.append(SEBlock(x.shape[1])) # After Layer 3
            x = v.layer4(x)
            self.se_modules.append(SEBlock(x.shape[1])) # After Layer 4

        # --- Define the new classification head ---
        proj_dim = v.output_dim
        hidden_dim = proj_dim // 2
        self.head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_cls)
        )

    def forward(self, images):
        v = self.clip.visual
        x = images.type(v.conv1.weight.dtype)
        
        # Manual forward pass through ResNet, inserting SE blocks
        # FIX: Replaced incorrect stem (with v.maxpool) with the correct ModifiedResNet stem from CLIP
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x)
        x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x)
        x = v.avgpool(x)
        # --- END FIX ---
        x = v.layer1(x)
        
        x = v.layer2(x)
        x = self.se_modules[0](x)
        
        x = v.layer3(x)
        x = self.se_modules[1](x)
        
        x = v.layer4(x)
        x = self.se_modules[2](x)
        
        # Final pooling and projection
        x = v.attnpool(x)
        
        return self.head(x)


# ---------------- Instantiate Model ----------------
model = CLIP_RN50_SENet(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)

# --- Collect trainable parameters (head + SE modules) ---
trainable_params = list(model.head.parameters()) + list(model.se_modules.parameters())

trainable_count = sum(p.numel() for p in trainable_params)
total_count = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_count:,} / {total_count:,} ({100.0*trainable_count/total_count:.2f}%)")


# ---------------- Optimizer / Loss / AMP / Scheduler ----------------
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ---------------- (Zero-shot and Evaluate functions are unchanged and correct) ----------------
@torch.no_grad()
def zero_shot_acc():
    clip_model.eval()
    templates = [f"a photo of a {c}" for c in train_ds.classes]
    text_tokens = clip.tokenize(templates).to(DEVICE)
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    correct = total = 0
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        image_features = clip_model.encode_image(imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds = sims.argmax(dim=-1)
        
        correct += (preds == labels.to(DEVICE)).sum().item()
        total += labels.size(0)
        
    return 100.0 * correct / total

@torch.no_grad()
def evaluate(net, print_report=False):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels, probs_list = [], [], []
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        with torch.amp.autocast(device_type=DEVICE):
            out = net(imgs)
            loss = criterion(out, labels)
            
        loss_sum += loss.item() * imgs.size(0)
        probs = torch.nn.functional.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        probs_list.append(probs.cpu().numpy())

    total = len(all_labels)
    avg_loss = loss_sum / total
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / total * 100.0
    probs_all = np.concatenate(probs_list, axis=0)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    logloss = log_loss(all_labels, probs_all, labels=list(range(NUM_CLASSES)))

    if print_report:
        print(f"Val loss: {avg_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=val_ds.classes, zero_division=0))

    return avg_loss, acc, prec, rec, f1, logloss


# ---------------- Training Loop ----------------
mlflow.set_experiment("clip_rn50_senet_probe") # <-- CHANGED
with mlflow.start_run():
    mlflow.log_params({
        "backbone": BACKBONE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "dropout": DROPOUT, "label_smoothing": LABEL_SMOOTHING,
        "architecture": "Frozen CLIP + SENet (L2,L3,L4) + MLP Head" # <-- CHANGED
    })

    zs_acc = zero_shot_acc()
    print(f"Zero-shot accuracy: {zs_acc:.2f}%")
    mlflow.log_metric("zero_shot_acc", zs_acc)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

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
        train_acc  = 100.0 * running_correct / running_samples
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_logloss = evaluate(model)
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

    # --- Final Evaluation ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation.")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        
    final_metrics = evaluate(model, print_report=True)
    mlflow.log_metrics({
        "final_val_loss": final_metrics[0], "final_val_acc": final_metrics[1],
        "final_val_precision": final_metrics[2], "final_val_recall": final_metrics[3],
        "final_val_f1": final_metrics[4], "final_val_logloss": final_metrics[5]
    })
    mlflow.pytorch.log_model(model, "clip_senet_head_model")

print("\nDone.")


