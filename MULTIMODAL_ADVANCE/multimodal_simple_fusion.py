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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import clip
from sklearn.metrics import f1_score
from PIL import Image
import warnings
import math

warnings.filterwarnings("ignore") 
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Set the desired GPU device


# --- GLOBAL CONFIGURATION ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WORKERS = 16 
DEFAULT_WORKERS = min(MAX_WORKERS, os.cpu_count() // 2) if os.cpu_count() else 4
PIN_MEMORY = True if DEVICE == "cuda" else False

BASE_DIR = Path("..") 
# Audio paths (features)
AUDIO_BASE_DIR = BASE_DIR / "ADVANCE_features"
AUDIO_TRAIN_DIR = AUDIO_BASE_DIR / "train"
AUDIO_TEST_DIR = AUDIO_BASE_DIR / "test"
NORMALIZER_PATH = AUDIO_BASE_DIR / "normalizer_train.npy"
# Vision paths (raw images)
VISION_BASE_DIR = BASE_DIR / "ADVANCE_DATA_split"
VISION_TRAIN_DIR = VISION_BASE_DIR / "train" / "vision"
VISION_TEST_DIR = VISION_BASE_DIR / "test" / "vision"
CHECKPOINT_PATH = "best_multimodal_clip_rn18_fusion.pt" 

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4 # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-2
PATIENCE = 15
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
CBAM_REDUCTION = 16 
SE_REDUCTION = 16


# --- LOAD AUDIO NORMALIZER ---
if not NORMALIZER_PATH.exists():
    raise FileNotFoundError(f"Normalizer file not found: {NORMALIZER_PATH}")
normalizer = np.load(NORMALIZER_PATH)
MU_AUDIO, SIGMA_AUDIO = normalizer[0], normalizer[1]


# --- CLIP Model Setup ---
CLIP_BACKBONE = "RN50"
print(f"Loading CLIP model {CLIP_BACKBONE} on {DEVICE}...")
clip_model, preprocess_val = clip.load(CLIP_BACKBONE, device=DEVICE, jit=False)
clip_model.eval()
IMAGE_SIZE = clip_model.visual.input_resolution
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# --- DATA TRANSFORMS ---
# Audio Transforms (Resizing)
audio_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
])
# Vision Transforms (CLIP-style with Augmentation for training)
train_vision_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
# Vision Transforms (CLIP standard preprocess for validation)
val_vision_transform = preprocess_val

# ==================== ATTENTION MODULES (SENet/CBAM) ====================

class SEBlock(nn.Module):
    # Squeeze-and-Excitation Block (Channel Attention only)
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.excitation(y)
        y = y.view(B, C, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    # Channel Attention Module (part of CBAM)
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    # Spatial Attention Module (part of CBAM)
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    # Convolutional Block Attention Module (CBAM)
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, reduction=reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        channel_refined_feature = x * self.ca(x)
        spatial_refined_feature = channel_refined_feature * self.sa(channel_refined_feature)
        return spatial_refined_feature

# ==================== AUDIO/VISION BACKBONES ====================

class SpecAugment(nn.Module):
    # SpecAugment for audio spectrograms
    def __init__(self, freq_mask_param=16, time_mask_param=50, num_freq_masks=1, num_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def forward(self, x):
        if not self.training:
            return x
        clone = x.clone()
        B, C, H, W = clone.shape
        # Frequency Masking (dim H)
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_param))
            if H - f > 0 and f > 0:
                f0 = int(np.random.uniform(0, H - f))
                clone[:, :, f0:f0 + f, :] = 0
        # Time Masking (dim W)
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_param))
            if W - t > 0 and t > 0:
                t0 = int(np.random.uniform(0, W - t))
                clone[:, :, :, t0:t0 + t] = 0
        return clone


class ResNet18_SENet_L4Only_Extractor(nn.Module):
    # ResNet18 with SENet after L4, modified to return features (before FC)
    def __init__(self, reduction=16):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Copy standard ResNet layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.se4 = SEBlock(512, reduction=reduction)
        self.avgpool = base_model.avgpool
        self.output_dim = base_model.fc.in_features # 512
        
        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze SEBlock
        for param in self.se4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        x = self.se4(x) # SENet only after layer4
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Feature vector of size 512
        return x


class CLIP_RN50_CBAM_Extractor(nn.Module):
    # CLIP ResNet50 with CBAM, modified to return features (before Head)
    def __init__(self, clip_model, cbam_reduction=16):
        super().__init__()
        self.clip_visual = clip_model.visual
        
        # Freeze CLIP visual backbone
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        self.cbam = CBAM(2048, reduction=cbam_reduction) # Output of layer4 is 2048
        self.output_dim = self.clip_visual.output_dim # 1024
        
        # Only unfreeze the CBAM block
        for param in self.cbam.parameters():
            param.requires_grad = True

    def forward(self, images):
        v = self.clip_visual
        x = images.type(v.conv1.weight.dtype)
        
        # Manual forward pass to inject CBAM
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
        
        x = v.layer4(x)
        x = self.cbam(x) # CBAM Injection
        
        # Final Attention Pooling (Standard CLIP output)
        x = v.attnpool(x) # Feature vector of size 1024
        return x.float()


# ==================== DATASET CLASSES ====================

class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, mu, sigma, transform=None):
        self.transform = transform
        self.mu = mu
        self.sigma = sigma
        self.root = Path(root_dir)
        self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples = self._find_samples()
    
    def _find_samples(self):
        samples = {} # Store as dictionary {file_basename: (feature_path, label_idx)}
        for class_name in self.class_names:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]
            for feature_path in class_dir.glob('*.npy'):
                # Use the file name (without extension) as the unique ID for multimodal pairing
                file_id = feature_path.stem
                samples[file_id] = (feature_path, label)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, file_id):
        feature_path, label = self.samples[file_id]
        sound = np.load(feature_path)
        sound = (sound - self.mu) / self.sigma
        sound = torch.from_numpy(sound).float()
        if sound.ndim > 2:
            sound = sound.squeeze()
        sound = sound.unsqueeze(0)
        sound = sound.expand(3, -1, -1) # Convert to 3-channel for pre-trained ResNet
        if self.transform:
            sound = self.transform(sound)
        return sound, label, file_id # Return file_id for inspection/debugging

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
        samples = {} # Store as dictionary {file_basename: (image_path, label_idx)}
        for class_name in self.classes:
            class_dir = self.root / class_name
            label = self.class_to_idx[class_name]
            if class_dir.is_dir():
                for image_path in class_dir.glob('*.jpg'):
                    # Use the file name (without extension) as the unique ID for multimodal pairing
                    file_id = image_path.stem
                    samples[file_id] = (image_path, label)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, file_id):
        image_path, label = self.samples[file_id]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, file_id # Return file_id for inspection/debugging

class MultimodalDataset(Dataset):
    """
    Combines audio and vision datasets based on matching file IDs (stem).
    Assumes matching file IDs exist in both audio_dir and vision_dir.
    """
    def __init__(self, audio_root_dir, vision_root_dir, mu_audio, sigma_audio, 
                 audio_transform, vision_transform):
        
        # 1. Initialize sub-datasets
        self.audio_ds = OfflineAudioDataset(audio_root_dir, mu_audio, sigma_audio, audio_transform)
        self.vision_ds = ImageFolderDataset(vision_root_dir, vision_transform)
        
        # Verify class correspondence
        if self.audio_ds.class_to_idx != self.vision_ds.class_to_idx:
             # This is a critical check for multimodal alignment
             raise ValueError("Class indices must match between audio and vision datasets!")

        self.class_names = self.audio_ds.class_names
        self.class_to_idx = self.audio_ds.class_to_idx
        
        # 2. Find common samples (Multimodal Samples)
        audio_ids = set(self.audio_ds.samples.keys())
        vision_ids = set(self.vision_ds.samples.keys())
        common_ids = sorted(list(audio_ids.intersection(vision_ids)))
        
        if not common_ids:
             raise RuntimeError(f"Found no matching samples between {audio_root_dir} ({len(audio_ids)} audio files) and {vision_root_dir} ({len(vision_ids)} image files). Check file naming convention.")

        # 3. Create final sample list (list of file_ids)
        self.multimodal_samples = common_ids
        
        # 4. Get labels for weighted sampling
        # We assume the label from the audio dataset is correct for all common samples.
        self.labels = [self.audio_ds.samples[file_id][1] for file_id in self.multimodal_samples]


    def __len__(self):
        return len(self.multimodal_samples)

    def __getitem__(self, index):
        file_id = self.multimodal_samples[index]
        
        # Retrieve data from sub-datasets using the file_id
        audio_data, label_a, _ = self.audio_ds[file_id]
        vision_data, label_v, _ = self.vision_ds[file_id]
        
        # Sanity check (should be covered by __init__ but good practice)
        assert label_a == label_v, "Mismatched labels for paired audio/vision data!"
        
        return audio_data, vision_data, label_a

# ==================== MULTIMODAL MODEL ====================

class MultimodalClassifier(nn.Module):
    def __init__(self, audio_extractor, vision_extractor, num_classes, dropout_rate=0.5):
        super().__init__()
        
        self.audio_extractor = audio_extractor # Output: 512
        self.vision_extractor = vision_extractor # Output: 1024
        
        # Downscale Audio feature from 1024 to 512 (though RN18 output is 512, this is a placeholder/option)
        # Note: RN18_SENet_L4Only_Extractor already outputs 512. 
        # For RN18 (512) and RN50 (1024), we make them compatible for fusion.
        # If both feature dimensions were different (e.g., 512 and 1024), a projection is needed.
        # Here, let's target a final feature size of 512 for both after modification/projection.
        
        # Vision feature is 1024. We need to downscale it to 512 for a simpler concatenation/fusion.
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_extractor.output_dim, 512), # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        # Audio feature is 512 (RN18 output) - no further projection needed, just dropout/BN
        self.audio_head = nn.Sequential(
            nn.BatchNorm1d(audio_extractor.output_dim), # 512
            nn.Dropout(p=dropout_rate)
        )

        # Fusion: Concatenate (512 + 512 = 1024)
        FUSION_DIM = 512 + 512 
        HIDDEN_DIM = FUSION_DIM // 2 # 512
        
        # Final Classification Head (Trainable)
        self.classifier_head = nn.Sequential(
            nn.Linear(FUSION_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(HIDDEN_DIM, num_classes)
        )
        
        # Check trainable parameters
        trainable_params = []
        trainable_params.extend(list(self.audio_extractor.se4.parameters()))
        trainable_params.extend(list(self.vision_extractor.cbam.parameters()))
        trainable_params.extend(list(self.vision_projection.parameters()))
        trainable_params.extend(list(self.audio_head.parameters()))
        trainable_params.extend(list(self.classifier_head.parameters()))

        print(f"Total Trainable Parameters (SE+CBAM+Fusion+Head): {sum(p.numel() for p in trainable_params):,}")


    def forward(self, audio, vision):
        # Extract features
        audio_features = self.audio_extractor(audio) # [B, 512]
        vision_features = self.vision_extractor(vision) # [B, 1024]
        
        # Project and prepare features for fusion
        audio_features_proj = self.audio_head(audio_features) # [B, 512]
        vision_features_proj = self.vision_projection(vision_features) # [B, 512]
        
        # Concatenate features
        fused_features = torch.cat([audio_features_proj, vision_features_proj], dim=1) # [B, 1024]
        
        # Classify
        out = self.classifier_head(fused_features)
        return out


# ==================== EVALUATION/TRAINING LOOP ====================

@torch.no_grad()
def evaluate(data_loader, net, criterion, device, set_name="Validation"):
    net.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels = [], []
    for audio_imgs, vision_imgs, labels in data_loader:
        audio_imgs, vision_imgs, labels = audio_imgs.to(device), vision_imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
            out = net(audio_imgs, vision_imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * audio_imgs.size(0)
        preds = out.argmax(dim=1)
        total += audio_imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = loss_sum / total
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / total
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1


if __name__ == '__main__':
    # 1. Dataset Loading
    train_ds = MultimodalDataset(
        audio_root_dir=AUDIO_TRAIN_DIR, 
        vision_root_dir=VISION_TRAIN_DIR, 
        mu_audio=MU_AUDIO, sigma_audio=SIGMA_AUDIO, 
        audio_transform=audio_transform, 
        vision_transform=train_vision_transform
    )
    val_ds = MultimodalDataset(
        audio_root_dir=AUDIO_TEST_DIR, 
        vision_root_dir=VISION_TEST_DIR, 
        mu_audio=MU_AUDIO, sigma_audio=SIGMA_AUDIO, 
        audio_transform=audio_transform, 
        vision_transform=val_vision_transform
    )

    class_names = train_ds.class_names
    NUM_CLASSES = len(class_names)
    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CLASSES} classes: {', '.join(class_names)}")
    print(f"Train samples: {len(train_ds)} | Test samples: {len(val_ds)}\n")

    # 2. Weighted Sampling Setup (Audio labels are used for consistency)
    class_counts = Counter(train_ds.labels)
    sample_weights = [1.0 / class_counts[label] for label in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    
    print(f"Using {DEFAULT_WORKERS} workers with pin_memory={PIN_MEMORY}.")
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=DEFAULT_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 3. Model and Optimization Setup
    audio_extractor = ResNet18_SENet_L4Only_Extractor(reduction=SE_REDUCTION).to(DEVICE)
    vision_extractor = CLIP_RN50_CBAM_Extractor(clip_model, cbam_reduction=CBAM_REDUCTION).to(DEVICE)
    model = MultimodalClassifier(audio_extractor, vision_extractor, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)

    # SpecAugment for audio only
    spec_augmenter = SpecAugment().to(DEVICE)

    # Only optimize the trainable parts (SEBlock, CBAM, Fusion/Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_f1 = 0.0
    patience_counter = 0

    # 4. Training Loop
    print("\n--- Starting Multimodal Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        spec_augmenter.train() 
        running_loss, running_correct, running_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for audio_imgs, vision_imgs, labels in pbar:
            audio_imgs, vision_imgs, labels = audio_imgs.to(DEVICE), vision_imgs.to(DEVICE), labels.to(DEVICE)
            
            # Apply SpecAugment during training to audio
            augmented_audio_imgs = spec_augmenter(audio_imgs) 
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
                out = model(augmented_audio_imgs, vision_imgs)
                loss = criterion(out, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * audio_imgs.size(0)
            preds = out.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_samples += audio_imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_correct/running_samples:.2f}%")
        
        train_loss = running_loss / running_samples
        train_acc = 100.0 * running_correct / running_samples
        
        # Evaluate
        spec_augmenter.eval() # Disable SpecAugment for evaluation
        val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, DEVICE) 
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved improved checkpoint (test_f1={val_f1:.4f}) to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # 5. Final Evaluation
    print("\n--- Training Complete ---")
    if Path(CHECKPOINT_PATH).exists():
        print("\n--- Evaluating Best Model on TEST Set ---")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        test_loss, test_acc, test_f1 = evaluate(val_loader, model, criterion, DEVICE)
        print(f"\nFinal Metrics on TEST Set:\nLoss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1-Score: {test_f1:.4f}")
    else:
        print("No checkpoint found to evaluate.")

    print("\n✅ Done.")