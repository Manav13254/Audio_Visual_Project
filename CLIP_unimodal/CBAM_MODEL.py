# cbam_model.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class ChannelAttn(nn.Module):
    """
    Channel attention as used in CBAM.
    Input: x (B, C, H, W)
    If return_map=True returns (out, ca_map) where ca_map has shape (B, C, 1, 1)
    """
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        hidden = max(1, channels // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_map: bool = False):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        ca_map = self.sig(a + m)  # (B, C, 1, 1)
        out = x * ca_map
        if return_map:
            return out, ca_map
        return out


class SpatialAttn(nn.Module):
    """
    Spatial attention as used in CBAM.
    Input: x (B, C, H, W)
    If return_map=True returns (out, sa_map) where sa_map has shape (B, 1, H, W)
    """
    def __init__(self, k: int = 7):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=pad, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_map: bool = False):
        # channel-wise avg and max -> concat -> conv -> sigmoid
        avg = torch.mean(x, dim=1, keepdim=True)
        _max = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, _max], dim=1)
        sa_map = self.sig(self.conv(cat))  # (B, 1, H, W)
        out = x * sa_map
        if return_map:
            return out, sa_map
        return out


class CBAM(nn.Module):
    """
    CBAM module wrapper combining Channel + Spatial attention.
    If return_maps=True returns (out, ca_map, sa_map)
    """
    def __init__(self, channels: int, r: int = 16, k: int = 7):
        super().__init__()
        self.ca = ChannelAttn(channels, r=r)
        self.sa = SpatialAttn(k=k)

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        if return_maps:
            x_ca, ca_map = self.ca(x, return_map=True)
            x_sa, sa_map = self.sa(x_ca, return_map=True)
            return x_sa, ca_map, sa_map
        else:
            x = self.ca(x)
            x = self.sa(x)
            return x


class CLIP_RN50_CBAM(nn.Module):
    """
    CLIP RN50 backbone with CBAM applied on final conv feature map.

    Constructor:
        CLIP_RN50_CBAM(clip_model, n_cls, num_cbam_layers=1, image_size=224, freeze_backbone=True, device="cpu")

    Arguments:
        - clip_model: the loaded CLIP model (from `clip.load("RN50", device=...)`)
        - n_cls: number of output classes
        - num_cbam_layers: how many CBAM blocks to stack (>=1)
        - image_size: input image size used to probe conv shapes
        - freeze_backbone: whether to freeze visual backbone params by default
        - device: torch device used for dummy forward when building (cpu/cuda)
    """
    def __init__(
        self,
        clip_model,
        n_cls: int,
        num_cbam_layers: int = 1,
        image_size: int = 224,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.clip = clip_model
        self.image_size = image_size

        if freeze_backbone:
            for p in self.clip.visual.parameters():
                p.requires_grad = False

        # compute channels by a single dummy forward through the resnet feature extractor
        dummy = torch.zeros(1, 3, image_size, image_size).to(device)
        with torch.no_grad():
            feat = self._resnet_feature_map(dummy)
        channels = feat.shape[1]

        # CBAM layers (stackable)
        self.cbams = nn.ModuleList([CBAM(channels) for _ in range(max(1, num_cbam_layers))])

        # classification head: mirror how CLIP's RN returns features
        proj_dim = getattr(self.clip.visual, "output_dim", None) or self.clip.visual.attnpool.output_dim
        self.head = nn.Linear(proj_dim, n_cls)

    def _resnet_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through CLIP's ModifiedResNet until the last convmap output.
        This mirrors the code in the training script.
        """
        v = self.clip.visual
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
        return x

    def forward(self, images: torch.Tensor, return_attn: bool = False
                ) -> torch.Tensor:
        """
        Forward pass.

        If return_attn == False:
            returns: out (B, n_cls)

        If return_attn == True:
            returns: (out, channel_maps, spatial_maps)
                - channel_maps: list of torch tensors, each (B, C, 1, 1) for each CBAM layer
                - spatial_maps: list of torch tensors, each (B, 1, H, W) for each CBAM layer

        NOTE: channel_maps and spatial_maps are the sigmoid attention maps (before being applied).
        """
        feat_map = self._resnet_feature_map(images)   # (B, C, H, W)
        x = feat_map

        channel_maps: List[torch.Tensor] = []
        spatial_maps: List[torch.Tensor] = []

        # apply CBAM layers and optionally collect maps
        if return_attn:
            for cbam in self.cbams:
                x, ca_map, sa_map = cbam(x, return_maps=True)
                channel_maps.append(ca_map)
                spatial_maps.append(sa_map)
        else:
            for cbam in self.cbams:
                x = cbam(x)

        v = self.clip.visual
        # if attnpool exists (as in CLIP RN), pass conv map to it
        if hasattr(v, "attnpool"):
            pooled = v.attnpool(x)   # attnpool handles flatten/proj inside CLIP
        else:
            pooled = x.mean(dim=[2, 3])
            if hasattr(v, "proj") and v.proj is not None:
                pooled = pooled @ v.proj

        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        out = self.head(pooled)

        if return_attn:
            return out, channel_maps, spatial_maps
        return out


def build_from_clip(clip_model, num_classes: int, num_cbam_layers: int = 1,
                    image_size: int = 224, freeze_backbone: bool = True, device: Optional[str] = None):
    """
    Convenience builder if you already loaded a clip model object.
    Example:
        import clip
        clip_model, _ = clip.load("RN50", device=device)
        model = build_from_clip(clip_model, num_classes=13, num_cbam_layers=1, image_size=224, freeze_backbone=True, device=device)
    """
    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return CLIP_RN50_CBAM(clip_model, num_classes, num_cbam_layers=num_cbam_layers, image_size=image_size, freeze_backbone=freeze_backbone, device=_device)


# no top-level execution: safe to import
if __name__ == "__main__":
    # tiny quick sanity check that won't run heavy training
    print("cbam_model.py loaded as script. This module is import-safe.")
