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
IMAGE_BACKBONE_TYPE = "RN50"
AUDIO_BACKBONE_TYPE = "RN18"

BASE_DIR = Path(r"E:\Audio_Visual_Project\final_data_split_80_20")
AUDIO_FEATURES_DIR = Path(r"E:\Audio_Visual_Project\preprocessed_features_80_20_original")

TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

IMAGE_CHECKPOINT_PATH = "best_image_clip_baseline_checkpoint.pt"
AUDIO_CHECKPOINT_PATH = "best_finetuned_rn18_model(offline).pt"
FUSION_CHECKPOINT_PATH = "best_multimodal_fusion_checkpoint.pt"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 15
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

try:
    normalizer = np.load(AUDIO_FEATURES_DIR / "normalizer.npy")
    mu, sigma = normalizer[0], normalizer[1]
except FileNotFoundError:
    print("Error: audio normalizer.npy not found. Please run the feature extraction script first.")
    exit()

clip_model, preprocess = clip.load(IMAGE_BACKBONE_TYPE, device=DEVICE, jit=False)
clip_model.eval()
IMAGE_SIZE = clip_model.visual.input_resolution
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

image_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
image_val_transform = preprocess

audio_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultimodalFolderDataset(Dataset):
    def __init__(self, root_dir, audio_features_dir, image_transform=None, audio_transform=None):
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.vision_root = Path(root_dir) / "vision"
        self.audio_features_root = Path(audio_features_dir)
        self.classes = sorted([d.name for d in self.vision_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_name in self.classes:
            class_vision_dir = self.vision_root / class_name
            for image_path in class_vision_dir.glob('*.jpg'):
                base_name = image_path.stem.split('_')[0]
                audio_feature_path = self.audio_features_root / class_name / f"{base_name}.npy"
                if audio_feature_path.exists():
                    label = self.class_to_idx[class_name]
                    samples.append((str(image_path), str(audio_feature_path), label))
        return [s for s in samples if s is not None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, audio_feature_path, label = self.samples[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        sound = np.load(audio_feature_path)
        sound = (sound - mu) / sigma
        sound = torch.from_numpy(sound).unsqueeze(0)
        sound = sound.expand(3, -1, -1)
        if self.audio_transform:
            sound = self.audio_transform(sound)
            
        return image, sound, label

class ImageFeatureExtractor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_visual = clip_model.visual
    def forward(self, images):
        x = images.type(self.clip_visual.conv1.weight.dtype)
        return self.clip_visual(x)

class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18()
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2), nn.ReLU(),
            nn.Dropout(p=0.7), nn.Linear(num_ftrs // 2, num_classes)
        )
    def forward(self, x):
        return self.backbone.fc[0](self.backbone(x))

class MultimodalFusionNet(nn.Module):
    def __init__(self, image_backbone, audio_backbone, num_classes, dropout_rate=0.5):
        super().__init__()
        self.image_backbone = image_backbone
        self.audio_backbone = audio_backbone
        
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        for param in self.audio_backbone.parameters():
            param.requires_grad = False
            
        image_feature_dim = self.image_backbone.clip_visual.output_dim
        audio_feature_dim = self.audio_backbone.backbone.fc.in_features // 2
        
        fusion_dim = image_feature_dim + audio_feature_dim
        hidden_dim = fusion_dim // 2
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, image, audio):
        image_features = self.image_backbone(image)
        audio_features = self.audio_backbone(audio)
        combined = torch.cat([image_features, audio_features], dim=1)
        return self.fusion_head(combined.float())

@torch.no_grad()
def evaluate(net, loader, criterion, class_names, print_report=False):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels = [], []
    for imgs, audios, labels in loader:
        imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE):
            out = net(imgs, audios)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = loss_sum / len(all_labels) if len(all_labels) > 0 else 0
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0 if len(all_labels) > 0 else 0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    if print_report:
        print(f"\n--- Final Validation Report ---")
        print(f"Val loss: {avg_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    return avg_loss, acc, f1

if __name__ == '__main__':
    train_ds = MultimodalFolderDataset(TRAIN_DIR, AUDIO_FEATURES_DIR / "train", image_train_transform, audio_transform)
    val_ds = MultimodalFolderDataset(VAL_DIR, AUDIO_FEATURES_DIR / "val", image_val_transform, audio_transform)
    
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [s[2] for s in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    image_feature_extractor = ImageFeatureExtractor(clip_model).to(DEVICE)
    audio_feature_extractor = AudioFeatureExtractor(NUM_CLASSES).to(DEVICE)
    
    image_feature_extractor.load_state_dict(torch.load(IMAGE_CHECKPOINT_PATH), strict=False)
    audio_feature_extractor.load_state_dict(torch.load(AUDIO_CHECKPOINT_PATH), strict=False)

    model = MultimodalFusionNet(image_feature_extractor, audio_feature_extractor, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = model.fusion_head.parameters()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    mlflow.set_experiment("multimodal_fusion_tuning")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            for imgs, audios, labels in pbar:
                imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE):
                    out = model(imgs, audios)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                running_corrects += (out.argmax(dim=1) == labels).sum().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0
            train_acc = 100.0 * running_corrects / len(train_ds) if len(train_ds) > 0 else 0
            
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
                torch.save(model.state_dict(), FUSION_CHECKPOINT_PATH)
                print(f"   -> Saved improved checkpoint (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"   -> No improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
        if os.path.exists(FUSION_CHECKPOINT_PATH):
            print(f"\nLoading best model for final evaluation.")
            model.load_state_dict(torch.load(FUSION_CHECKPOINT_PATH, map_location=DEVICE))
            final_loss, final_acc, final_f1 = evaluate(model, val_loader, criterion, train_ds.classes, print_report=True)
            mlflow.log_metrics({
                "final_val_loss": final_loss, "final_val_acc": final_acc, "final_val_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "multimodal_fusion_model")

    print("\n✅ Done.")
