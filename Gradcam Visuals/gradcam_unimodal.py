import os
import json   # <-- ADDED
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

# Where we save the sampled image ids
SELECTED_IDS_PATH = "selected_ids.json"   # <-- ADDED

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
test_transform = preprocess


# ====================== DATASET ======================
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
                for image_path in class_dir.glob("*.jpg"):
                    label = self.class_to_idx[class_name]
                    samples.append((str(image_path), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label


# ====================== CBAM MODULES ======================
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
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
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_c = torch.mean(x, dim=1, keepdim=True)
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_c, max_c], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ====================== CLIP + CBAM HEAD ======================
class CLIP_RN50_CBAM_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5, cbam_reduction=16):
        super().__init__()
        self.clip_visual = clip_model.visual

        for p in self.clip_visual.parameters():
            p.requires_grad = False

        self.cbam = CBAM(2048, reduction=cbam_reduction)

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

        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x)
        x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x)
        x = v.avgpool(x)

        x = v.layer1(x)
        x = v.layer2(x)
        x = v.layer3(x)
        x = v.layer4(x)

        x = self.cbam(x)
        x = v.attnpool(x)

        return self.head(x.float())


@torch.no_grad()
def evaluate(net, loader, criterion, class_names):
    net.eval()
    loss_sum = 0.0
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE=="cuda")):
            out = net(imgs)
            loss = criterion(out, labels)

        loss_sum += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = loss_sum / len(labels_all)
    acc = (np.array(preds_all) == np.array(labels_all)).mean() * 100
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

    return avg_loss, acc, f1


# ====================== GRAD-CAM ======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.act = None
        self.grad = None

        self.layer.register_forward_hook(self.fwd)
        self.layer.register_full_backward_hook(self.bwd)

    def fwd(self, m, inp, out):
        self.act = out.detach()

    def bwd(self, m, gin, gout):
        self.grad = gout[0].detach()

    def __call__(self, inp, cls_idx):
        self.model.zero_grad()
        inp.requires_grad_(True)

        out = self.model(inp)
        one_hot = torch.zeros_like(out)
        one_hot[:, cls_idx] = 1

        (one_hot * out).sum().backward(retain_graph=True)

        A = self.act
        G = self.grad
        w = G.mean(dim=(2,3), keepdim=True)

        cam = (A * w).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inp.shape[-2:], mode="bicubic")
        cam = cam[0,0].cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam


def show_cam_on_image(img, heatmap):
    H, W, _ = img.shape
    heat = Image.fromarray((heatmap*255).astype(np.uint8)).resize((W,H))
    heat = np.array(heat) / 255.0
    return heat


# ====================== UPDATED visualize_grad_cam (SAVES JSON!) ======================
def visualize_grad_cam(model, dataset, classes_to_visualize, samples_per_class=5):

    # Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.to(DEVICE).float().eval()
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print(f"Checkpoint missing → {CHECKPOINT_PATH}")
        return

    # Grad-CAM layer
    cam_layer = model.cbam
    cam = GradCAM(model, cam_layer)

    # Sample tracking for JSON
    selected_ids = {}   # <-- NEW

    class_to_idx = dataset.class_to_idx

    # Collect samples
    target_samples = []
    for cls in classes_to_visualize:

        cls_id = class_to_idx[cls]
        cls_samples = [s for s in dataset.samples if s[1] == cls_id]

        chosen = random.sample(cls_samples, samples_per_class)
        target_samples.extend([(p, l, cls) for p, l in chosen])

        # Save only file IDs (stem)
        selected_ids[cls] = [Path(p).stem for p,_ in chosen]   # <-- NEW

    # SAVE JSON FILE HERE
    with open(SELECTED_IDS_PATH, "w") as f:
        json.dump(selected_ids, f, indent=4)

    print(f"\nSaved selected samples → {SELECTED_IDS_PATH}\n")
    print(selected_ids)

    # Plotting
    unique_classes = list(selected_ids.keys())
    rows = samples_per_class
    cols = len(unique_classes)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    cmap = plt.cm.jet
    plot_idx = {c:0 for c in unique_classes}

    for path, lbl, clsname in target_samples:

        c = unique_classes.index(clsname)
        r = plot_idx[clsname]

        img_pil = Image.open(path).convert("RGB")
        inp = test_transform(img_pil).unsqueeze(0).to(DEVICE).float()

        heat = cam(inp, lbl)

        img_np = np.array(img_pil)/255.0
        heat_resized = show_cam_on_image(img_np, heat)

        ax = axes[r,c]
        ax.imshow(img_np)
        ax.imshow(heat_resized, cmap=cmap, alpha=0.5)
        ax.set_title(clsname)
        ax.axis("off")

        plot_idx[clsname] += 1

    plt.tight_layout()
    plt.show()



# ====================== MAIN ======================
if __name__ == "__main__":

    train_ds = ImageFolderDataset(TRAIN_DIR, transform=train_transform)
    test_ds  = ImageFolderDataset(TEST_DIR, transform=test_transform)

    model = CLIP_RN50_CBAM_Head(
        clip_model,
        num_classes=len(train_ds.classes),
        dropout_rate=DROPOUT,
        cbam_reduction=CBAM_REDUCTION
    ).to(DEVICE)

    TARGET_CLASSES = ["beach", "bridge", "sports land"]
    SAMPLES_PER_CLASS = 5

    visualize_grad_cam(model, test_ds, TARGET_CLASSES, SAMPLES_PER_CLASS)

    print("\n✔ Script finished.\n")
