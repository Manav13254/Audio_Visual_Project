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
import clip
from sklearn.metrics import f1_score
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_BACKBONE_TYPE = "RN50"

IMAGE_DATA_DIR = Path("../data/ADVANCE_images_split")
AUDIO_DATA_DIR = Path("../data/processed_audio")
TRAIN_IMAGE_DIR = IMAGE_DATA_DIR / "train"
VAL_IMAGE_DIR = IMAGE_DATA_DIR / "val"
TEST_IMAGE_DIR = IMAGE_DATA_DIR / "test"
TRAIN_AUDIO_DIR = AUDIO_DATA_DIR / "train"
VAL_AUDIO_DIR = AUDIO_DATA_DIR / "val"
TEST_AUDIO_DIR = AUDIO_DATA_DIR / "test"

IMAGE_CHECKPOINT_PATH = "best_image_clip_senet_70-10-20.pt"
AUDIO_CHECKPOINT_PATH = "bam_resnet18_512.pth"
FUSION_CHECKPOINT_PATH = "best_multimodal_cross_attention.pt"

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 15
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

clip_model, preprocess = clip.load(IMAGE_BACKBONE_TYPE, device="cpu", jit=False)
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
audio_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True)])

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

            # Ensure both class directories exist before searching
            if not image_class_dir.is_dir() or not audio_class_dir.is_dir():
                continue

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

        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

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
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return self.mlp(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)

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

class CLIP_RN50_SE_Head(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.se2 = SELayer(512)
        self.se3 = SELayer(1024)
        self.se4 = SELayer(2048)
        output_dim = self.clip_visual.output_dim
        hidden_dim = output_dim // 2
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, images):
        v = self.clip_visual; x = images.type(v.conv1.weight.dtype)
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x); x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x); x = v.avgpool(x)
        x = v.layer1(x); x = v.layer2(x); x = self.se2(x); x = v.layer3(x)
        x = self.se3(x); x = v.layer4(x); x = self.se4(x); x = v.attnpool(x)
        return self.head(x.float())

class ImageFeatureExtractor(nn.Module):
    def __init__(self, clip_model, num_classes, checkpoint_path):
        super().__init__()
        self.full_model = CLIP_RN50_SE_Head(clip_model, num_classes)
        self.full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self.feature_extractor_head = nn.Sequential(*list(self.full_model.head.children())[:-1])
        self.output_dim = 512 

    def forward(self, images):
        v = self.full_model.clip_visual; x = images.type(v.conv1.weight.dtype)
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x); x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x); x = v.avgpool(x)
        x = v.layer1(x); x = v.layer2(x); x = self.full_model.se2(x); x = v.layer3(x)
        x = self.full_model.se3(x); x = v.layer4(x); x = self.full_model.se4(x)
        features_pre_head = v.attnpool(x).float()
        
        return self.feature_extractor_head(features_pre_head)

class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_classes, checkpoint_path):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.layer2.add_module("BAM", BAM(128))
        backbone.layer3.add_module("BAM", BAM(256))
        backbone.layer4.add_module("BAM", BAM(512))
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )
        backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.projector = nn.Sequential(*list(backbone.fc.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.projector(x)

# --- FUSION MODEL ---
class CrossAttentionFusionNet(nn.Module):
    def __init__(self, image_backbone, audio_backbone, num_classes, dropout_rate=0.5):
        super().__init__()
        self.image_backbone = image_backbone
        self.audio_backbone = audio_backbone
        
        for param in self.image_backbone.parameters(): param.requires_grad = False
        for param in self.audio_backbone.parameters(): param.requires_grad = False

        image_feature_dim = self.image_backbone.output_dim # 512
        audio_feature_dim = self.audio_backbone.output_dim # 512
        common_embed_dim = 512
        num_attention_heads = 8
        
        self.image_proj = nn.Linear(image_feature_dim, common_embed_dim)
        self.audio_proj = nn.Linear(audio_feature_dim, common_embed_dim)
        
        self.audio_cross_attention = nn.MultiheadAttention(embed_dim=common_embed_dim, num_heads=num_attention_heads, batch_first=True)
        self.image_cross_attention = nn.MultiheadAttention(embed_dim=common_embed_dim, num_heads=num_attention_heads, batch_first=True)
        
        fusion_dim = common_embed_dim * 2
        hidden_dim = fusion_dim // 2
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, audio):
        with torch.no_grad():
            image_features = self.image_backbone(image)
            audio_features = self.audio_backbone(audio)
        
        image_embed = self.image_proj(image_features).unsqueeze(1)
        audio_embed = self.audio_proj(audio_features).unsqueeze(1)
        
        audio_aware_image_feat, _ = self.audio_cross_attention(query=audio_embed, key=image_embed, value=image_embed)
        image_aware_audio_feat, _ = self.image_cross_attention(query=image_embed, key=audio_embed, value=audio_embed)
        
        combined = torch.cat([audio_aware_image_feat.squeeze(1), image_aware_audio_feat.squeeze(1)], dim=1)
        return self.fusion_head(combined)

@torch.no_grad()
def evaluate(net, loader, criterion):
    net.eval()
    loss_sum, all_preds, all_labels = 0.0, [], []
    for (imgs, audios), labels in loader: # Corrected unpacking
        imgs, audios, labels = imgs.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
            out = net(imgs, audios)
            if criterion:
                loss = criterion(out, labels)
                loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = loss_sum / len(all_labels) if all_labels and criterion else 0
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels) * 100.0 if all_labels else 0
    return avg_loss, acc, f1

if __name__ == '__main__':
    train_ds = MultimodalDataset(TRAIN_IMAGE_DIR, TRAIN_AUDIO_DIR, image_train_transform, audio_transform)
    val_ds = MultimodalDataset(VAL_IMAGE_DIR, VAL_AUDIO_DIR, image_val_transform, audio_transform)
    
    if len(train_ds) == 0:
        print("FATAL ERROR: Training dataset is empty. Check paths and data matching logic."); exit()
        
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    train_labels = [s[2] for s in train_ds.samples]
    class_counts = Counter(train_labels)
    sampler = WeightedRandomSampler([1.0/class_counts[l] for l in train_labels], num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing and loading pre-trained unimodal feature extractors...")
    image_feature_extractor = ImageFeatureExtractor(clip_model, NUM_CLASSES, IMAGE_CHECKPOINT_PATH).to(DEVICE)
    audio_feature_extractor = AudioFeatureExtractor(NUM_CLASSES, AUDIO_CHECKPOINT_PATH).to(DEVICE)
    print("Extractors loaded successfully.")
    
    model = CrossAttentionFusionNet(image_feature_extractor, audio_feature_extractor, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    
    trainable_params = list(model.image_proj.parameters()) + list(model.audio_proj.parameters()) + list(model.audio_cross_attention.parameters()) + list(model.image_cross_attention.parameters()) + list(model.fusion_head.parameters())
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for (imgs, sounds), labels in pbar: 
            imgs, sounds, labels = imgs.to(DEVICE), sounds.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True) 
            
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
                out = model(imgs, sounds)
                loss = criterion(out, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss = running_loss / len(train_ds)
        
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), FUSION_CHECKPOINT_PATH)
            print(f"  -> Saved improved checkpoint (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f" -> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break
            
    print("\n--- Evaluating Best Model on TEST Set ---")
    if os.path.exists(FUSION_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(FUSION_CHECKPOINT_PATH, map_location=DEVICE))
        
        test_ds = MultimodalDataset(TEST_IMAGE_DIR, TEST_AUDIO_DIR, image_val_transform, audio_transform)
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            print(f"Test samples: {len(test_ds)}")
            final_loss, final_acc, final_f1 = evaluate(model, test_loader, criterion)
            
            print("\n--- Final TEST Set Report ---")
            print(f"Test Loss: {final_loss:.4f} | Test Acc: {final_acc:.2f}% | Test F1: {final_f1:.4f}")
        else:
            print("Test dataset is empty. Could not perform final evaluation.")

    print("\n✅ Done.")