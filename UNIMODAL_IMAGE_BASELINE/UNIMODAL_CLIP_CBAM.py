import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import clip
from sklearn.metrics import f1_score, classification_report
from PIL import Image
from collections import Counter
import warnings
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# --- CONFIGURATION ---
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

BASE_DIR = Path("../ADVANCE_DATA_split") 
TRAIN_DIR = BASE_DIR / "train" / "vision"
TEST_DIR = BASE_DIR / "test" / "vision"
CHECKPOINT_PATH = "best_image_clip_rn50_cbam_advancesplit.pt" 
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
CBAM_REDUCTION = 16

# --- CRITICAL CLIP LOADING FIX ---
print(f"Loading CLIP model {BACKBONE} on CPU (as float32) for stable Grad-CAM...")
# Load model weights onto CPU first (defaulting to float32)
clip_model, preprocess = clip.load(BACKBONE, device="cpu", jit=False) 

# Convert model to float32 and move to the target device
clip_model = clip_model.float() 
clip_model.to(DEVICE)
clip_model.eval()
# ---------------------------------

IMAGE_SIZE = clip_model.visual.input_resolution
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
test_transform = preprocess

# --- DATASET CLASS ---
class ImageFolderDataset(Dataset):
    """Loads images from a standard directory structure: root/class_name/image.jpg"""
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
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, label

# --- MODEL COMPONENTS: CBAM ---

class ChannelAttention(nn.Module):
    """Channel Attention Module (part of CBAM)"""
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
    """Spatial Attention Module (part of CBAM)"""
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
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 1. Channel Attention
        channel_refined_feature = x * self.ca(x)
        # 2. Spatial Attention
        spatial_refined_feature = channel_refined_feature * self.sa(channel_refined_feature)
        return spatial_refined_feature


class CLIP_RN50_CBAM_Head(nn.Module):
    """CLIP ResNet50 with CBAM integrated after the last ResNet block."""
    def __init__(self, clip_model, num_classes, dropout_rate=0.5, cbam_reduction=16):
        super().__init__()
        self.clip_visual = clip_model.visual
        
        # Freeze CLIP visual backbone
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        # CBAM block (Trainable) - 2048 channels for RN50 layer4 output
        self.cbam = CBAM(2048, reduction=cbam_reduction) 
        
        output_dim = self.clip_visual.output_dim # 1024
        hidden_dim = output_dim // 2
        
        # Trainable Classification Head
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        v = self.clip_visual
        # FIX: Force the input tensor to be float32 for Grad-CAM stability
        x = images.type(torch.float32) 
        
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
        
        # Integrate CBAM after layer4
        x = v.layer4(x)
        x = self.cbam(x) # <--- CBAM Injection
        
        # Final Attention Pooling (Standard CLIP output)
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

# --- GRAD-CAM IMPLEMENTATION ---

class GradCAM:
    """Computes Grad-CAM for a PyTorch model."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Register hooks
        self.hook_handles.append(target_layer.register_forward_hook(self._forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_category: Optional[int] = None) -> torch.Tensor:
        self.model.zero_grad()
        
        # 1. Forward pass
        input_tensor.requires_grad_(True) 
        output = self.model(input_tensor)
        
        if target_category is None:
            target_category = output.argmax(dim=1).item()
        
        # 2. Backward pass: Calculate gradient of target class score w.r.t model outputs
        one_hot = torch.zeros_like(output)
        one_hot[:, target_category] = 1.0
        target_output = torch.sum(output * one_hot)
        
        target_output.backward(retain_graph=True) 
        input_tensor.requires_grad_(False) 

        # 3. Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of Gradients (weights)
        weights = F.adaptive_avg_pool2d(gradients, 1)
        
        # Weighted combination of Activation maps (CAM)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # 4. Interpolate and Normalize
        cam = F.interpolate(cam, 
                            size=input_tensor.shape[-2:], 
                            mode='bicubic', 
                            align_corners=False)
        
        cam = cam.squeeze(1) # Shape: (B, H, W)
        
        # Per-sample normalization
        B, H, W = cam.shape
        for i in range(B):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max - cam_min > 1e-5:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = torch.zeros_like(cam[i])
                
        return cam.cpu()

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

# --- VISUALIZATION HELPER (Only returns the resized heatmap) ---
def show_cam_on_image(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Returns the heatmap (NumPy array) resized to the image dimensions (0-1 range)."""
    
    H, W, _ = img.shape
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.array(heatmap_resized) / 255.0

def visualize_grad_cam(model, dataset, classes_to_visualize: List[str], samples_per_class: int = 5):
    """Generates and displays Grad-CAM visualizations for target classes using Matplotlib overlay."""
    
    # 1. Load the best checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        
        # The model was already set to float32 upon initialization, but we ensure it here again 
        # for maximum safety, though the fix is primarily in the CLIP loading.
        model.float()
        
        model.to(DEVICE)
        model.eval()
        print(f"Loaded model for Grad-CAM from {CHECKPOINT_PATH}.")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Cannot perform Grad-CAM.")
        return

    # 2. Identify the target layer (The CBAM output feature map)
    target_layer = model.cbam
    cam_generator = GradCAM(model, target_layer)
    print(f"Grad-CAM hook registered on layer: {target_layer.__class__.__name__}")

    # 3. Collect target image paths and labels
    target_samples_info = [] # (path, label, class_name)
    class_to_idx = dataset.class_to_idx
    
    for class_name in classes_to_visualize:
        if class_name not in class_to_idx:
            print(f"Warning: Class '{class_name}' not found. Please check folder name in {TEST_DIR}. Skipping.")
            continue
            
        target_idx = class_to_idx[class_name]
        class_samples = [s for s in dataset.samples if s[1] == target_idx]
        
        if len(class_samples) > samples_per_class:
            selected_samples = random.sample(class_samples, samples_per_class)
        else:
            selected_samples = class_samples
            
        target_samples_info.extend([(path, label, class_name) for path, label in selected_samples])

    if not target_samples_info:
        print("No samples found for visualization.")
        return

    # 4. Generate and plot Grad-CAMs
    print(f"\nGenerating Grad-CAMs for {len(target_samples_info)} samples...")
    
    unique_classes = sorted(list(set(info[2] for info in target_samples_info)))
    rows = max(samples_per_class, 1)
    cols = len(unique_classes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    plot_indices = {name: 0 for name in unique_classes}
    cmap = plt.cm.jet # Use standard jet colormap

    for image_path_str, target_label, class_name in target_samples_info:
        col_idx = unique_classes.index(class_name)
        row_idx = plot_indices[class_name]
        
        if row_idx >= rows: continue

        # Load original image and create input tensor
        original_img = Image.open(image_path_str).convert('RGB')
        
        # 1. Apply transform and move to device
        input_tensor = test_transform(original_img).unsqueeze(0).to(DEVICE)
        
        # 2. Input casting (CRITICAL)
        input_tensor = input_tensor.to(torch.float32) 
        
        rgb_img_np = np.array(original_img)
        
        # Generate Grad-CAM 
        try:
            grayscale_cam = cam_generator(input_tensor, target_category=target_label)[0].numpy()
        except Exception as e:
            print(f"Warning: Failed to generate CAM for {Path(image_path_str).name}. Error: {e}")
            grayscale_cam = np.zeros(rgb_img_np.shape[:2])

        # Resize heatmap array (0-1 range)
        heatmap_resized_float = show_cam_on_image(rgb_img_np, grayscale_cam)

        # Plotting
        ax = axes[row_idx, col_idx]
        
        # Display the original image (normalized to 0-1 for display)
        ax.imshow(rgb_img_np / 255.0) 
        
        # Overlay the heatmap using the 'jet' colormap and transparency (alpha=0.5)
        if np.max(heatmap_resized_float) > 1e-6:
             ax.imshow(heatmap_resized_float, 
                       cmap=cmap, 
                       alpha=0.5, 
                       interpolation='nearest') 
        
        ax.set_title(f"Label: {class_name}", fontsize=10)
        ax.axis('off')
        
        plot_indices[class_name] += 1
        
    # Final plot cleanup
    for col_idx, class_name in enumerate(unique_classes):
        axes[0, col_idx].set_title(f"**{class_name}** (Grad-CAM)", fontsize=12, color='blue')
        
    for row_idx in range(rows):
        if cols > 0:
             axes[row_idx, 0].text(-0.1, 0.5, f"Sample {row_idx+1}", 
                              transform=axes[row_idx, 0].transAxes, 
                              rotation=90, va='center', ha='right', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.suptitle("Grad-CAM Visualization for CLIP-RN50-CBAM Fine-Tuning", y=1.0, fontsize=16, fontweight='bold')
    plt.show()

# --- MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    # --- 1. Setup and Train/Load Model ---
    try:
        train_ds = ImageFolderDataset(TRAIN_DIR, transform=train_transform)
        test_ds = ImageFolderDataset(TEST_DIR, transform=test_transform)
    except FileNotFoundError as e:
        print(f"Error: Data directory not found. Please ensure {BASE_DIR} exists and is populated.")
        print(f"Details: {e}")
        exit()
        
    NUM_CLASSES = len(train_ds.classes)
    print(f"Found {NUM_CLASSES} classes: {train_ds.classes}")

    # Model initialization uses the globally loaded clip_model (which is now float32)
    model = CLIP_RN50_CBAM_Head(clip_model, NUM_CLASSES, dropout_rate=DROPOUT, cbam_reduction=CBAM_REDUCTION).to(DEVICE)
    
    # --- Training/Loading Logic ---
    if not os.path.exists(CHECKPOINT_PATH):
        print("\n--- Checkpoint not found. Starting Training/Fine-Tuning ---")
        
        train_labels = [item[1] for item in train_ds.samples]
        class_counts = Counter(train_labels)
        sample_weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_test_f1 = 0.0
        patience_counter = 0
        
        # ... (Training loop logic here) ...
        # NOTE: You MUST re-run training once to save a new checkpoint with float32 weights!
        
        # Simplified placeholder logic to ensure model loads for Grad-CAM
        if os.path.exists(CHECKPOINT_PATH):
             model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
             print("\nNote: Training skipped/completed. Running final evaluation/Grad-CAM.")
        else:
             print("\nNote: Please run the training loop once to generate the checkpoint.")


    # --- 2. Grad-CAM Visualization ---
    TARGET_CLASSES = ["beach", "bridge", "sports club"] 
    SAMPLES_PER_CLASS = 5

    visualize_grad_cam(model, test_ds, TARGET_CLASSES, SAMPLES_PER_CLASS)
    
    print("\n✅ Script execution complete.")