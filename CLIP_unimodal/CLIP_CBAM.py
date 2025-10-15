
import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
BACKBONE = "RN50"

# <<< CHANGED: Updated paths to match our 70/10/20 split >>>
BASE_DIR = Path("/home/23ucs671/audio_visual_proj1/ADVANCE_images_split")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test" # <<< ADDED: Path for the final test set
CHECKPOINT_PATH = "best_image_clip_cbam_70-10-20.pt"

# <<< Hyperparameters from your script (unchanged) >>>
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

# --- MODEL COMPONENTS (Unchanged) ---
import torch
import torch.nn as nn

# --- CBAM Components  ---
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
        # The attention is applied to the input feature map
        x = x * self.ca(x) 
        x = x * self.sa(x)
        return x

# --- Corrected CLIP Head with CBAM Blocks using Hooks ---
class CLIP_RN50_CBAM_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        # Freeze the original backbone
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        # Define your trainable CBAM Layers
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        # --- REGISTER HOOKS ---
        # This tells PyTorch to apply CBAM after the original layer computation
        self.clip_visual.layer2.register_forward_hook(self.get_hook(self.cbam2))
        self.clip_visual.layer3.register_forward_hook(self.get_hook(self.cbam3))
        self.clip_visual.layer4.register_forward_hook(self.get_hook(self.cbam4))

        # Define your trainable classification head
        output_dim = self.clip_visual.output_dim
        hidden_dim = output_dim // 2
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    # This function creates the hook
    def get_hook(self, cbam_layer):
        def hook(model, input, output):
            # Apply the CBAM block to the output of the original layer
            return cbam_layer(output)
        return hook
    
    # This function creates the hook
    def get_hook(self, cbam_layer):
        def hook(model, input, output):
            # 1. Store the original data type (e.g., float16)
            original_dtype = output.dtype
            
            # 2. Convert to float32 for the custom CBAM layer
            output_float32 = output.float()
            processed_output = cbam_layer(output_float32)
            
            # 3. Convert back to the original type before returning
            return processed_output.to(original_dtype)
        return hook

    def forward(self, images):
        # The forward pass is simple and safe again!
        # Hooks automatically apply your CBAM layers at the right spots.
        x = self.clip_visual(images)
        return self.head(x.float())

# --- EVALUATION FUNCTION (Unchanged) ---
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
    val_ds = ImageFolderDataset(VAL_DIR, transform=val_transform)
    
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [item[1] for item in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

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

    model = CLIP_RN50_CBAM_Head(clip_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = list(model.head.parameters()) + list(model.cbam2.parameters()) + \
                     list(model.cbam3.parameters()) + list(model.cbam4.parameters())
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    mlflow.set_experiment("image_clip_cbam_70-10-20")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

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
        print("\n--- Evaluating Best Model on TEST Set ---")
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

            # Create test dataloader
            test_ds = ImageFolderDataset(TEST_DIR, transform=val_transform)
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
            )
            
            print(f"Test samples: {len(test_ds)}")
            
            final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion, test_ds.classes)
            
            print("\n--- Final TEST Set Report ---")
            print(f"Test Loss: {final_loss:.4f} | Test Acc: {final_acc:.2f}% | Test F1: {final_f1:.4f}")
            
            # Get full classification report on the test set
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(DEVICE)
                    out = model(imgs)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            print(classification_report(
                all_labels,
                all_preds,
                target_names=test_ds.classes,
                zero_division=0
            ))

            mlflow.log_metrics({
                "final_test_loss": final_loss, "final_test_acc": final_acc, "final_test_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "image_clip_cbam_model_final")

    print("\n✅ Done.")