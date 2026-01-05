# compare_random5_side_by_side.py
import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
import clip
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
SEED = None  # None -> fresh random every run
random.seed(SEED)
np.random.seed(SEED)
if SEED is not None:
    torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path("..")
AUDIO_BASE_DIR = BASE_DIR / "ADVANCE_features"
AUDIO_TEST_DIR  = AUDIO_BASE_DIR / "test"
VISION_BASE_DIR = BASE_DIR / "ADVANCE_DATA_split"
VISION_TEST_DIR = VISION_BASE_DIR / "test" / "vision"

NORMALIZER_PATH = AUDIO_BASE_DIR / "normalizer_train.npy"
UNIMODAL_CHECKPOINT = "best_image_clip_rn50_cbam_advancesplit.pt"
MULTIMODAL_CHECKPOINT = "best_multimodal_clip_rn18_cab_fusion.pt"

CBAM_REDUCTION = 16
SE_REDUCTION = 16

OUT_DIR = Path("outputs/side_by_side")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 5

# -------------------------
# Load normalizer & CLIP
# -------------------------
if not NORMALIZER_PATH.exists():
    raise FileNotFoundError(f"Missing normalizer: {NORMALIZER_PATH}")
normalizer = np.load(NORMALIZER_PATH)
MU_AUDIO, SIGMA_AUDIO = normalizer[0], normalizer[1]

CLIP_BACKBONE = "RN50"
print(f"Loading CLIP {CLIP_BACKBONE} on {DEVICE}...")
clip_model, preprocess_val = clip.load(CLIP_BACKBONE, device=DEVICE, jit=False)
clip_model.eval()

val_vision_transform = preprocess_val
audio_transform = transforms.Compose([transforms.Resize((224,224), antialias=True)])

# -------------------------
# CBAM / SE / backbones / models (copied from your code)
# -------------------------
class SEBlock(nn.Module):
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
        y = self.excitation(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

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
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction=reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        channel_refined_feature = x * self.ca(x)
        spatial_refined_feature = channel_refined_feature * self.sa(channel_refined_feature)
        return spatial_refined_feature

class CLIP_RN50_CBAM_Head(nn.Module):
    def __init__(self, clip_model, num_classes, dropout_rate=0.5, cbam_reduction=16):
        super().__init__()
        self.clip_visual = clip_model.visual
        for p in self.clip_visual.parameters(): p.requires_grad = False
        self.cbam = CBAM(2048, reduction=cbam_reduction)
        output_dim = self.clip_visual.output_dim
        hidden = output_dim // 2
        self.head = nn.Sequential(nn.Linear(output_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(hidden, num_classes))
    def forward(self, images):
        v = self.clip_visual
        x = images.type(v.conv1.weight.dtype)
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x)
        x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x)
        x = v.avgpool(x)
        x = v.layer1(x); x = v.layer2(x); x = v.layer3(x); x = v.layer4(x)
        x = self.cbam(x)
        x = v.attnpool(x)
        return self.head(x.float())

class CLIP_RN50_CBAM_Extractor(nn.Module):
    def __init__(self, clip_model, cbam_reduction=16):
        super().__init__()
        self.clip_visual = clip_model.visual
        for p in self.clip_visual.parameters(): p.requires_grad = False
        self.cbam = CBAM(2048, reduction=cbam_reduction)
        self.output_dim = self.clip_visual.output_dim
    def forward(self, images, return_map=False):
        v = self.clip_visual
        x = images.type(v.conv1.weight.dtype)
        x = v.conv1(x); x = v.bn1(x); x = v.relu1(x)
        x = v.conv2(x); x = v.bn2(x); x = v.relu2(x)
        x = v.conv3(x); x = v.bn3(x); x = v.relu3(x)
        x = v.avgpool(x)
        x = v.layer1(x); x = v.layer2(x); x = v.layer3(x); x = v.layer4(x)
        x_cbam = self.cbam(x)
        pooled = v.attnpool(x_cbam)
        if return_map:
            return pooled.float(), x_cbam.detach()
        return pooled.float()

class ResNet18_SENet_L4Only_Extractor(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = base.conv1; self.bn1 = base.bn1; self.relu = base.relu; self.maxpool = base.maxpool
        self.layer1 = base.layer1; self.layer2 = base.layer2; self.layer3 = base.layer3; self.layer4 = base.layer4
        self.se4 = SEBlock(512, reduction=reduction)
        self.avgpool = base.avgpool
        self.output_dim = base.fc.in_features
        for p in self.parameters(): p.requires_grad = False
        for p in self.se4.parameters(): p.requires_grad = True
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.se4(x)
        x = self.avgpool(x)
        return torch.flatten(x,1)

# Datasets
class OfflineAudioDataset(Dataset):
    def __init__(self, root_dir, mu, sigma, transform=None):
        self.transform = transform; self.mu = mu; self.sigma = sigma; self.root = Path(root_dir)
        self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {name:i for i,name in enumerate(self.class_names)}
        self.samples = self._find_samples()
    def _find_samples(self):
        samples = {}
        for cname in self.class_names:
            cdir = self.root / cname; label = self.class_to_idx[cname]
            for f in cdir.glob("*.npy"):
                samples[f.stem] = (f, label)
        return samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, file_id):
        path,label = self.samples[file_id]
        arr = np.load(path); arr = (arr - self.mu)/self.sigma
        t = torch.from_numpy(arr).float()
        if t.ndim>2: t = t.squeeze()
        t = t.unsqueeze(0).expand(3,-1,-1)
        if self.transform: t = self.transform(t)
        return t, label, file_id

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform; self.root = Path(root_dir)
        if not self.root.exists(): raise FileNotFoundError(self.root)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = self._find_samples()
    def _find_samples(self):
        s = {}
        for cname in self.classes:
            cdir = self.root / cname; label = self.class_to_idx[cname]
            if cdir.is_dir():
                for p in cdir.glob("*.jpg"):
                    s[p.stem] = (str(p), label)
        return s
    def __len__(self): return len(self.samples)
    def __getitem__(self, file_id):
        p,label = self.samples[file_id]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label, file_id

class MultimodalDataset(Dataset):
    def __init__(self, audio_root_dir, vision_root_dir, mu, sigma, audio_transform, vision_transform):
        self.audio_ds = OfflineAudioDataset(audio_root_dir, mu, sigma, audio_transform)
        self.vision_ds = ImageFolderDataset(vision_root_dir, vision_transform)
        if self.audio_ds.class_to_idx != self.vision_ds.class_to_idx:
            raise ValueError("class mismatch")
        self.class_names = self.audio_ds.class_names
        self.class_to_idx = self.audio_ds.class_to_idx
        audio_ids = set(self.audio_ds.samples.keys()); vision_ids = set(self.vision_ds.samples.keys())
        common = sorted(list(audio_ids.intersection(vision_ids)))
        if not common: raise RuntimeError("no common multimodal ids")
        self.multimodal_samples = common
        self.labels = [self.audio_ds.samples[fid][1] for fid in self.multimodal_samples]
    def __len__(self): return len(self.multimodal_samples)
    def __getitem__(self, idx):
        fid = self.multimodal_samples[idx]
        a,la,_ = self.audio_ds[fid]; v,lv,_ = self.vision_ds[fid]
        assert la==lv
        return a,v,la

# Multimodal model
class ModalityAttentionGenerator(nn.Module):
    def __init__(self, feature_dim, dropout_rate=0.5):
        super().__init__()
        self.mask_generator = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Dropout(p=dropout_rate), nn.Sigmoid())
    def forward(self,x): return self.mask_generator(x)

class MultimodalClassifier_CAB(nn.Module):
    def __init__(self, audio_extractor, vision_extractor, num_classes, dropout_rate=0.5):
        super().__init__()
        self.audio_extractor = audio_extractor
        self.vision_extractor = vision_extractor
        self.vision_projection = nn.Sequential(nn.Linear(vision_extractor.output_dim,512), nn.ReLU(inplace=True), nn.Dropout(p=dropout_rate))
        self.audio_head = nn.Sequential(nn.BatchNorm1d(audio_extractor.output_dim), nn.Dropout(p=dropout_rate))
        FEATURE_DIM = 512
        self.vision_mask_generator = ModalityAttentionGenerator(FEATURE_DIM, dropout_rate)
        self.audio_mask_generator  = ModalityAttentionGenerator(FEATURE_DIM, dropout_rate)
        FUSED_DIM = FEATURE_DIM*2; HIDDEN_DIM = FUSED_DIM//2
        self.classifier_head = nn.Sequential(nn.Linear(FUSED_DIM,HIDDEN_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(HIDDEN_DIM,num_classes))
    def forward(self,audio,vision):
        A_raw = self.audio_extractor(audio); V_raw = self.vision_extractor(vision)
        A_proj = self.audio_head(A_raw); V_proj = self.vision_projection(V_raw)
        M_V = self.vision_mask_generator(V_proj); M_A = self.audio_mask_generator(A_proj)
        A_ref = A_proj + (A_proj * M_V); V_ref = V_proj + (V_proj * M_A)
        fused = torch.cat([A_ref, V_ref], dim=1)
        return self.classifier_head(fused)

# -------------------------
# Grad-CAM wrappers (unimodal & multimodal) hooking CBAM
# -------------------------
class GradCAMUnimodal:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.act = None
        self.grad = None
        self.handles = []
        self.handles.append(self.layer.register_forward_hook(self._fwd))
        try:
            self.handles.append(self.layer.register_full_backward_hook(self._bwd))
        except Exception:
            self.handles.append(self.layer.register_backward_hook(self._bwd))
    def _fwd(self, m, i, o): self.act = o.detach()
    def _bwd(self, m, gi, go): self.grad = go[0].detach()
    def remove(self):
        for h in self.handles:
            try: h.remove()
            except: pass
    def __call__(self, inp_tensor, target_class=None):
        self.model.zero_grad()
        inp = inp_tensor.clone().detach().requires_grad_(True)
        logits = self.model(inp)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        one_hot = torch.zeros_like(logits); one_hot[:, target_class] = 1.0
        (logits * one_hot).sum().backward(retain_graph=False)
        A = self.act; G = self.grad
        if A is None or G is None:
            raise RuntimeError("hooks not fired (unimodal)")
        w = G.mean(dim=(2,3), keepdim=True)
        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inp.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        mn,mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx-mn+1e-9) if mx>mn else np.zeros_like(cam)
        return cam

class GradCAMMultimodal:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.act = None
        self.grad = None
        self.handles = []
        self.handles.append(self.layer.register_forward_hook(self._fwd))
        try:
            self.handles.append(self.layer.register_full_backward_hook(self._bwd))
        except Exception:
            self.handles.append(self.layer.register_backward_hook(self._bwd))
    def _fwd(self, m, i, o): self.act = o.detach()
    def _bwd(self, m, gi, go): self.grad = go[0].detach()
    def remove(self):
        for h in self.handles:
            try: h.remove()
            except: pass
    def __call__(self, audio_tensor, vision_tensor, target_class=None):
        self.model.zero_grad()
        v = vision_tensor.clone().detach().requires_grad_(True)
        a = audio_tensor.clone().detach()
        logits = self.model(a.to(DEVICE), v.to(DEVICE))
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        one_hot = torch.zeros_like(logits); one_hot[0, target_class] = 1.0
        (logits * one_hot).sum().backward(retain_graph=False)
        A = self.act; G = self.grad
        if A is None or G is None:
            raise RuntimeError("hooks not fired (multimodal)")
        w = G.mean(dim=(2,3), keepdim=True)
        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=v.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
        mn,mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx-mn+1e-9) if mx>mn else np.zeros_like(cam)
        return cam

# -------------------------
# overlay helper (no text)
# -------------------------
def overlay_on_image(pil_img, heatmap, alpha=0.45):
    img_np = np.array(pil_img).astype(np.float32)/255.0
    heat_rgb = plt.cm.jet(heatmap)[:,:,:3]
    combined = (1-alpha)*img_np + alpha*heat_rgb
    combined = np.clip(combined, 0,1)
    return (combined*255).astype(np.uint8)

# -------------------------
# MAIN
# -------------------------
def main():
    # Build dataset
    val_ds = MultimodalDataset(str(AUDIO_TEST_DIR), str(VISION_TEST_DIR), MU_AUDIO, SIGMA_AUDIO, audio_transform, val_vision_transform)
    print(f"Multimodal dataset loaded: {len(val_ds)} samples, {len(val_ds.class_names)} classes")

    # Map class -> list of multimodal IDs
    class_to_ids = defaultdict(list)
    for fid, lbl in zip(val_ds.multimodal_samples, val_ds.labels):
        class_to_ids[lbl].append(fid)
    # map idx->name
    idx_to_name = {v:k for k,v in val_ds.class_to_idx.items()}

    # pick random samples per class
    chosen = {}
    for class_idx, ids in class_to_ids.items():
        if len(ids) == 0:
            chosen[class_idx] = []
        else:
            if len(ids) <= SAMPLES_PER_CLASS:
                chosen[class_idx] = ids.copy()
            else:
                chosen[class_idx] = random.sample(ids, SAMPLES_PER_CLASS)
    print("Random selection per class complete.")

    # Build models
    num_classes = len(val_ds.class_names)
    unimodal_model = CLIP_RN50_CBAM_Head(clip_model, num_classes, cbam_reduction=CBAM_REDUCTION).to(DEVICE)
    multimodal_model = MultimodalClassifier_CAB(ResNet18_SENet_L4Only_Extractor(SE_REDUCTION).to(DEVICE),
                                                CLIP_RN50_CBAM_Extractor(clip_model, cbam_reduction=CBAM_REDUCTION).to(DEVICE),
                                                num_classes, dropout_rate=0.5).to(DEVICE)

    # load checkpoints if available (optional)
    if Path(UNIMODAL_CHECKPOINT).exists():
        unimodal_model.load_state_dict(torch.load(UNIMODAL_CHECKPOINT, map_location=DEVICE))
        print("Loaded unimodal checkpoint.")
    else:
        print("Unimodal checkpoint not found; running with current weights.")

    if Path(MULTIMODAL_CHECKPOINT).exists():
        multimodal_model.load_state_dict(torch.load(MULTIMODAL_CHECKPOINT, map_location=DEVICE))
        print("Loaded multimodal checkpoint.")
    else:
        print("Multimodal checkpoint not found; running with current weights.")

    unimodal_model.to(DEVICE).float().eval()
    multimodal_model.to(DEVICE).float().eval()

    # CAM hooks on CBAM (option B)
    unimodal_cam = GradCAMUnimodal(unimodal_model, unimodal_model.cbam)
    multimodal_cam = GradCAMMultimodal(multimodal_model, multimodal_model.vision_extractor.cbam)

    # Iterate and save side-by-side
    for class_idx, files in chosen.items():
        class_name = idx_to_name[class_idx]
        for fid in files:
            try:
                idx = val_ds.multimodal_samples.index(fid)
            except ValueError:
                print(f"Warning: {fid} missing from multimodal_samples; skipping")
                continue

            audio_t, vision_t, label = val_ds[idx]
            audio_input = audio_t.unsqueeze(0).to(DEVICE).float()
            vision_input = vision_t.unsqueeze(0).to(DEVICE).float()

            # compute unimodal CAM
            try:
                cam_uni = unimodal_cam(vision_input, target_class=label)
            except Exception as e:
                print(f"Unimodal CAM error for {fid}: {e}")
                cam_uni = np.zeros((vision_input.shape[-2], vision_input.shape[-1]))

            # compute multimodal CAM
            try:
                cam_multi = multimodal_cam(audio_input, vision_input, target_class=label)
            except Exception as e:
                print(f"Multimodal CAM error for {fid}: {e}")
                cam_multi = np.zeros((vision_input.shape[-2], vision_input.shape[-1]))

            # load original image and resize cams to original size
            img_path, _ = val_ds.vision_ds.samples[fid]
            pil_img = Image.open(img_path).convert("RGB")

            cam_uni_pil = Image.fromarray((cam_uni*255).astype(np.uint8))
            cam_multi_pil = Image.fromarray((cam_multi*255).astype(np.uint8))
            if cam_uni_pil.size != pil_img.size:
                cam_uni_pil = cam_uni_pil.resize(pil_img.size, resample=Image.BILINEAR)
            if cam_multi_pil.size != pil_img.size:
                cam_multi_pil = cam_multi_pil.resize(pil_img.size, resample=Image.BILINEAR)
            cam_uni_np = np.array(cam_uni_pil).astype(np.float32)/255.0
            cam_multi_np = np.array(cam_multi_pil).astype(np.float32)/255.0

            left = overlay_on_image(pil_img, cam_uni_np, alpha=0.45)
            right = overlay_on_image(pil_img, cam_multi_np, alpha=0.45)

            # side-by-side
            side = np.concatenate([left, right], axis=1)
            out_pil = Image.fromarray(side)
            safe_class = class_name.replace(" ", "_")
            out_name = OUT_DIR / f"{safe_class}_{fid}.png"
            out_pil.save(out_name)
            print(f"Saved: {out_name}")

    # cleanup
    unimodal_cam.remove(); multimodal_cam.remove()
    print("All done. Side-by-side images are in:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
