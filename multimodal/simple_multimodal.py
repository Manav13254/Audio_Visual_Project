import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# --- Basic Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import clip
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_BASE_DIR = Path("../data/ADVANCE_images_split")
AUDIO_BASE_DIR = Path("../data/processed_audio")

TRAIN_DIR_IMG = IMAGE_BASE_DIR / "train"
VAL_DIR_IMG = IMAGE_BASE_DIR / "val"
TEST_DIR_IMG = IMAGE_BASE_DIR / "test"

TRAIN_DIR_AUDIO = AUDIO_BASE_DIR / "train"
VAL_DIR_AUDIO = AUDIO_BASE_DIR / "val"
TEST_DIR_AUDIO = AUDIO_BASE_DIR / "test"

IMAGE_CHECKPOINT_PATH = "best_image_clip_senet_70-10-20.pt"
audio_full_model_path = "bam_resnet18_512.pth"
MULTIMODAL_CHECKPOINT_PATH = "best_multimodal_fusion.pt"

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5

_, preprocess = clip.load("RN50", device="cpu")
image_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])
image_val_transform = preprocess

audio_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
])

class MultimodalDataset(Dataset):
    def __init__(self, image_root_dir, audio_root_dir, image_transform=None, audio_transform=None):
        self.image_root = Path(image_root_dir)
        self.audio_root = Path(audio_root_dir)
        self.image_transform = image_transform
        self.audio_transform = audio_transform

        if not self.image_root.exists() or not self.audio_root.exists():
            raise FileNotFoundError(f"Image ({self.image_root}) or Audio ({self.audio_root}) directory not found.")

        self.classes = sorted([d.name for d in self.image_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        print(f"Searching for pairs in {self.image_root.name}...")
        for class_name in self.classes:
            image_class_dir = self.image_root / class_name
            audio_class_dir = self.audio_root / class_name
            class_idx = self.class_to_idx[class_name]

            for image_path in image_class_dir.glob('*.jpg'):
                audio_path = audio_class_dir / f"{image_path.stem}.npy"
                if audio_path.exists():
                    samples.append((image_path, audio_path, class_idx))
        print(f"Found {len(samples)} paired image/audio samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, audio_path, label = self.samples[idx]

        # Process Image
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # Process Audio
        audio_spec = np.load(audio_path)
        audio_tensor = torch.from_numpy(audio_spec).float()
        if audio_tensor.ndim > 2:
            audio_tensor = audio_tensor.squeeze()
        audio_tensor = audio_tensor.unsqueeze(0).expand(3, -1, -1)
        if self.audio_transform:
            audio_tensor = self.audio_transform(audio_tensor)

        return (image, audio_tensor), label

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size(); y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1); return x * y.expand_as(x)

class CLIP_RN50_SE_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.se2 = SELayer(512); self.se3 = SELayer(1024); self.se4 = SELayer(2048)
        output_dim = self.clip_visual.output_dim
        self.head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(output_dim // 2, num_classes)
        )
    def forward(self, images):
        v = self.clip_visual; x = images.type(v.conv1.weight.dtype)
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x); x = v.conv2(x); x = v.bn2(x)
        x = v.relu2(x); x = v.conv3(x); x = v.bn3(x); x = v.relu3(x); x = v.avgpool(x)
        x = v.layer1(x); x = v.layer2(x); x = self.se2(x); x = v.layer3(x)
        x = self.se3(x); x = v.layer4(x); x = self.se4(x); x = v.attnpool(x)
        return self.head(x.float())

class ImageFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super().__init__()
        clip_model_base, _ = clip.load("RN50", device="cpu", jit=False)
        self.base_model = CLIP_RN50_SE_Head(clip_model_base, num_classes)
        self.base_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        
        original_head = self.base_model.head
        self.feature_extractor_head = nn.Sequential(*list(original_head.children())[:-1])
        print("Image Feature Extractor loaded and head adapted.")

    def forward(self, x):
        v = self.base_model.clip_visual; img = x.type(v.conv1.weight.dtype)
        img = v.conv1(img); img = v.bn1(img); img = v.relu1(img); img = v.conv2(img); img = v.bn2(img)
        img = v.relu2(img); img = v.conv3(img); img = v.bn3(img); img = v.relu3(img); img = v.avgpool(img)
        img = v.layer1(img); img = v.layer2(img); img = self.base_model.se2(img); img = v.layer3(img)
        img = self.base_model.se3(img); img = v.layer4(img); img = self.base_model.se4(img); img = v.attnpool(img)
        features = self.feature_extractor_head(img.float())
        return features

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels)
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

class AudioFeatureExtractor(nn.Module):
    def __init__(self, full_model_checkpoint_path, num_classes):
        super().__init__()
        model = models.resnet18(weights=None)
        model.layer2.add_module("BAM", BAM(128))
        model.layer3.add_module("BAM", BAM(256))
        model.layer4.add_module("BAM", BAM(512))
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(),
            nn.Dropout(0.6), nn.Linear(512, num_classes)
        )
        
        full_model_state_dict = torch.load(full_model_checkpoint_path, map_location="cpu")
        model.load_state_dict(full_model_state_dict)
        print("Full audio model loaded successfully from:", full_model_checkpoint_path)

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_projector = nn.Sequential(*list(model.fc.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.feature_projector(x)
        return x

class MultimodalFusionModel(nn.Module):
    def __init__(self, image_extractor, audio_extractor, num_classes, dropout_rate=0.5):
        super().__init__()
        self.image_extractor = image_extractor
        self.audio_extractor = audio_extractor

        for param in self.image_extractor.parameters(): param.requires_grad = False
        for param in self.audio_extractor.parameters(): param.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, audios):
        image_features = self.image_extractor(images)
        audio_features = self.audio_extractor(audios)
        combined_features = torch.cat([image_features, audio_features], dim=1)
        output = self.fusion_head(combined_features)
        return output

@torch.no_grad()
def evaluate(net, loader, criterion):
    net.eval()
    loss_sum, all_preds, all_labels = 0.0, [], []
    for (imgs, audios), labels in loader:
        imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            out = net(imgs, audios)
            loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = loss_sum / len(all_labels)
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1

if __name__ == '__main__':
    train_ds = MultimodalDataset(TRAIN_DIR_IMG, TRAIN_DIR_AUDIO, image_train_transform, audio_transform)
    val_ds = MultimodalDataset(VAL_DIR_IMG, VAL_DIR_AUDIO, image_val_transform, audio_transform)
    
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [item[2] for item in train_ds.samples]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing feature extractors...")
    image_feat_extractor = ImageFeatureExtractor(IMAGE_CHECKPOINT_PATH, NUM_CLASSES)
    
    audio_feat_extractor = AudioFeatureExtractor(audio_full_model_path, NUM_CLASSES)
    
    model = MultimodalFusionModel(
        image_extractor=image_feat_extractor,
        audio_extractor=audio_feat_extractor,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.fusion_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"Training only the fusion head on {DEVICE}...")

    mlflow.set_experiment("multimodal_late_fusion_70-10-20")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in globals().items() if isinstance(v, (str, int, float)) and k.isupper()})

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss, running_corrects = 0.0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            for (imgs, audios), labels in pbar:
                imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    out = model(imgs, audios)
                    loss = criterion(out, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                running_corrects += (preds == labels).sum().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_ds)
            train_acc = 100.0 * running_corrects / len(train_ds)
            
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")

            if val_f1 > best_val_f1 + MIN_DELTA:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), MULTIMODAL_CHECKPOINT_PATH)
                print(f"   -> Saved improved checkpoint (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
                
        print("\n--- Evaluating Best Model on TEST Set ---")
        if os.path.exists(MULTIMODAL_CHECKPOINT_PATH):
            model.load_state_dict(torch.load(MULTIMODAL_CHECKPOINT_PATH, map_location=DEVICE))

            test_ds = MultimodalDataset(TEST_DIR_IMG, TEST_DIR_AUDIO, image_val_transform, audio_transform)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion)
            
            print("\n--- Final TEST Set Report ---")
            print(f"Test Loss: {final_loss:.4f} | Test Acc: {final_acc:.2f}% | Test F1: {final_f1:.4f}")
            
            mlflow.log_metrics({
                "final_test_loss": final_loss, "final_test_acc": final_acc, "final_test_f1": final_f1
            })
            mlflow.pytorch.log_model(model, "multimodal_fusion_model_final")

    print("\n✅ Done.") 