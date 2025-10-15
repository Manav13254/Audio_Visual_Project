
import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

# --- CONFIGURATION ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"

# <<< CHANGED: Updated paths to match our 70/10/20 split >>>
BASE_DIR = Path("/home/23ucs671/audio_visual_proj1/ADVANCE_images_split")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test" 
CHECKPOINT_PATH = "best_image_clip_rn50_70-10-20.pt"

# <<< CHANGED: Hyperparameters from your script (unchanged but listed for clarity) >>>
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

# --- CLIP MODEL LOADING (Unchanged) ---
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

# --- DATASET CLASS ---
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # <<< CHANGED: Removed the '/ "vision"' subfolder assumption >>>
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
            
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_name in self.classes:
            # <<< CHANGED: Path logic simplified >>>
            class_dir = self.root / class_name
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

# --- MODEL, EVALUATION, ZERO-SHOT (Unchanged) ---
class CLIP_RN50_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        for param in self.clip_visual.parameters():
            param.requires_grad = False
        
        output_dim = self.clip_visual.output_dim
        hidden_dim = output_dim // 2
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        x = images.type(self.clip_visual.conv1.weight.dtype)
        x = self.clip_visual(x)
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

@torch.no_grad()
def zero_shot_acc(loader, class_names):
    clip_model.eval()
    templates = [f"a photo of a {c.replace('_', ' ')}" for c in class_names] # Improved templates
    text_tokens = clip.tokenize(templates).to(DEVICE)
    with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
        text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            image_features = clip_model.encode_image(imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds = similarities.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return 100.0 * correct / total if total > 0 else 0.0

if __name__ == '__main__':
    train_ds = ImageFolderDataset(TRAIN_DIR, transform=train_transform)
    val_ds = ImageFolderDataset(VAL_DIR, transform=val_transform)
    
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [item[1] for item in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Use multiple CPU cores to load data in parallel
    NUM_WORKERS = 8 

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True  # Speeds up host (CPU) to device (GPU) transfers
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = CLIP_RN50_Head(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = model.head.parameters()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    mlflow.set_experiment("image_clip_baseline_70-10-20")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

        zs_acc = zero_shot_acc(val_loader, train_ds.classes)
        print(f"Zero-shot accuracy on validation set: {zs_acc:.2f}%")
        mlflow.log_metric("zero_shot_acc", zs_acc)

        best_val_f1 = 0.0
        patience_counter = 0

        # Training loop is unchanged
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
            
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, train_ds.classes)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, F1={train_f1:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")

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
                
        # <<< CHANGED: FINAL EVALUATION NOW RUNS ON THE UNSEEN TEST SET >>>
        # --- CORRECTED FINAL EVALUATION ---
        print("\n--- Evaluating Best Model on TEST Set ---")
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

            test_ds = ImageFolderDataset(TEST_DIR, transform=val_transform)
            # Ensure you add num_workers and pin_memory here
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
            )
            
            print(f"Test samples: {len(test_ds)}")
            
            # 1. Get predictions and labels by iterating through the test_loader
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in tqdm(test_loader, desc="Generating Test Predictions"):
                    imgs = imgs.to(DEVICE)
                    with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                        out = model(imgs)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            # 2. Now calculate metrics using the collected predictions and labels
            final_acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0
            final_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            
            print("\n--- Final TEST Set Report ---")
            print(f"Test Acc: {final_acc:.2f}% | Test F1: {final_f1:.4f}")
            print(classification_report(
                all_labels, 
                all_preds, 
                target_names=test_ds.classes,
                zero_division=0
            ))
            
            # You might want to log the final F1 score as well
            mlflow.log_metrics({
                "final_test_acc": final_acc, "final_test_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "image_clip_baseline_model_final")

    print("\n✅ Done.")