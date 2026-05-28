"""
TCDI-Net: Texture-Structure Complementary Dual-branch Interaction Network for SR-IQA

This implementation is aligned with the thesis Chapter 3 description and Fig. 3-2:

1) Input-level scale-controlled Gaussian decomposition
      I_S = G_{sigma,k} * I_SR,  I_T = I_SR - I_S
2) Dual-branch backbone
      image branch: I_SR -> [ResBlock + DCAB + TGB feedback] x 4
      detail branch: I_T  -> four-stage ResBlock features
3) TGB is applied stage by stage and added back to the image branch feature.
4) GDIB is applied once after the final image feature F^4, producing F_Low and F_High.
5) MSFFB fuses multi-scale image features F^1...F^4 to obtain F_Fusion.
6) BOP and heterogeneous aggregation:
      F_NR = [BOP(F_Fusion); mean(F_Low); var(F_High)]
7) Optional DR branch:
      I_LR -> bicubic -> I_BI; compare structure components of I_SR and I_BI with frozen VGG.
8) Unified Huber + PLCC training loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """Convolution with automatic same padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size - 1) * dilation // 2,
        bias=bias,
        stride=stride,
        dilation=dilation,
    )


def global_mean_var(x: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Channel-wise first-order mean and second-order variance."""
    mean = x.mean(dim=(2, 3))
    var = (x - mean[:, :, None, None]).pow(2).mean(dim=(2, 3))
    return mean, var.clamp_min(eps)


# -----------------------------------------------------------------------------
# Input-level scale-controlled decomposition: Eq. (3.1)-(3.3)
# -----------------------------------------------------------------------------

class GaussianDecomposition(nn.Module):
    """
    Scale-controlled base/detail decomposition.

    I_S = G_{sigma,k} * I_SR
    I_T = I_SR - I_S

    This is an operation-level base/detail split rather than a strict semantic
    separation of structure and texture.
    """

    def __init__(self, channels: int = 3, sigma: float = 1.2):
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        radius = kernel_size // 2

        ax = torch.arange(kernel_size, dtype=torch.float32) - radius
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        weight = gauss_2d[None, None, :, :].repeat(channels, 1, 1, 1)

        self.channels = channels
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = F.conv2d(
            x,
            self.weight,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )
        detail = x - base
        return base, detail


# -----------------------------------------------------------------------------
# DCAB: Dilated Channel Attention Block
# -----------------------------------------------------------------------------

class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 4, bias: bool = False):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class DCAB(nn.Module):
    """
    Dilated convolution + channel attention.
    Auxiliary enhancement module for multi-scale SR artifact modeling.
    """

    def __init__(
        self,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 4,
        bias: bool = False,
        dilation: int = 2,
    ):
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, bias=bias, dilation=1),
            nn.PReLU(),
            conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation),
        )
        self.ca = CALayer(n_feat, reduction=reduction, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


# -----------------------------------------------------------------------------
# ResBlock and stage modules
# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(out + identity)


class ImageStage(nn.Module):
    """
    One image-branch stage in Fig. 3-2: ResBlock + DCAB.
    Stage 1 uses a stem to downsample input image to H/4; later stages use ResBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, stage_index: int, use_dcab: bool = True):
        super().__init__()
        if stage_index == 0:
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif stage_index in (1, 2):
            self.extractor = ResBlock(in_channels, out_channels, stride=2)
        else:
            self.extractor = ResBlock(in_channels, out_channels, stride=1)

        self.use_dcab = use_dcab
        self.dcab = DCAB(out_channels) if use_dcab else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.extractor(x)
        if self.use_dcab:
            out = self.dcab(out)
        return out


class DetailBranch(nn.Module):
    """
    Detail branch for I_T.
    It extracts four-stage features used by TGB/GDIB as texture/detail conditions.

    Outputs:
      F_T^1: 64 channels,  H/4  x W/4
      F_T^2: 128 channels, H/8  x W/8
      F_T^3: 256 channels, H/16 x W/16
      F_T^4: 512 channels, H/16 x W/16
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = ResBlock(c1, c2, stride=2)
        self.stage3 = ResBlock(c2, c3, stride=2)
        self.stage4 = ResBlock(c3, c4, stride=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


# -----------------------------------------------------------------------------
# TGB: Texture Guided Block, Eq. (3.4)-(3.5)
# -----------------------------------------------------------------------------

class TGB(nn.Module):
    """
    Texture/detail-guided spatial reweighting.

    F_omega^i = Sigmoid(psi_i(I_T))
    F_TGB^i   = phi_i(F_I^i) * F_omega^i

    In the main network, F_TGB^i is added back to F_I^i and then passed to the
    next image stage, matching Fig. 3-2.
    """

    def __init__(self, img_channels: int, texture_channels: int = 3):
        super().__init__()
        self.psi = nn.Conv2d(texture_channels, img_channels, kernel_size=1, bias=True)
        self.phi = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, img_feat: torch.Tensor, detail_img: torch.Tensor) -> torch.Tensor:
        weight = self.psi(detail_img)
        if weight.shape[2:] != img_feat.shape[2:]:
            weight = F.interpolate(
                weight,
                size=img_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        weight = torch.sigmoid(weight)
        return self.phi(img_feat) * weight


# -----------------------------------------------------------------------------
# GDIB: Gated Dynamic Interaction Block, Eq. (3.6)-(3.11)
# -----------------------------------------------------------------------------

class GDIB(nn.Module):
    """
    Detail-conditioned dynamic low/high-frequency complementary modeling.

    l is predicted from the final detail feature F_T^4.
    h = K_delta - l, so F_High = F_I^4 - F_Low.
    This block is applied once after F^4, as shown in Fig. 3-2.
    """

    def __init__(self, det_channels: int, img_channels: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        if img_channels % groups != 0:
            raise ValueError(f"img_channels={img_channels} must be divisible by groups={groups}")

        self.groups = groups
        self.kernel_size = kernel_size
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.kernel_predictor = nn.Conv2d(
            det_channels,
            groups * kernel_size * kernel_size,
            kernel_size=1,
            bias=False,
        )
        self.gate = nn.Conv2d(
            groups * kernel_size * kernel_size,
            groups * kernel_size * kernel_size,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(groups * kernel_size * kernel_size)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(
        self,
        img_feat: torch.Tensor,
        det_feat: torch.Tensor,
        return_kernel: bool = False,
    ):
        b, c, h, w = img_feat.shape
        k2 = self.kernel_size * self.kernel_size

        # Eq. (3.6)-(3.8): detail-conditioned dynamic low-pass kernel
        kernel = self.kernel_predictor(self.gap(det_feat))
        kernel = kernel * torch.sigmoid(self.gate(kernel))
        kernel = self.bn(kernel)
        kernel = kernel.view(b, self.groups, k2, 1)
        kernel = F.softmax(kernel, dim=2)

        # Eq. (3.9): grouped dynamic low-pass filtering
        patches = F.unfold(self.pad(img_feat), kernel_size=self.kernel_size)
        patches = patches.view(b, self.groups, c // self.groups, k2, h * w)

        low = (patches * kernel[:, :, None, :, :]).sum(dim=3)
        low = low.view(b, c, h, w)

        # Eq. (3.10)-(3.11): high-pass complement
        high = img_feat - low

        if return_kernel:
            return low, high, kernel
        return low, high


# -----------------------------------------------------------------------------
# MSFFB: Multi-Scale Feature Fusion Block
# -----------------------------------------------------------------------------

class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.sigmoid(
            self.channel_mlp(self.avg_pool(x)) + self.channel_mlp(self.max_pool(x))
        )
        x = x * ca

        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        sa = self.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * sa


class MSFFB(nn.Module):
    """
    Multi-scale feature fusion block.
    It fuses image-branch features F^1, F^2, F^3 and F^4 to the deepest spatial size.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1, bias=True) for ch in in_channels_list]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attn = ChannelSpatialAttention(out_channels)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        target_size = feats[-1].shape[2:]
        projected = []

        for feat, proj in zip(feats, self.projs):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj(feat))

        out = torch.cat(projected, dim=1)
        return self.attn(self.fusion(out))


# -----------------------------------------------------------------------------
# DR extension: structure compensation branch, Eq. (3.16)-(3.23)
# -----------------------------------------------------------------------------

def build_vgg16_features(pretrained: bool = True) -> nn.Sequential:
    """
    Build VGG16 feature extractor up to conv4_3.

    The import is lazy to avoid breaking the whole IQANet definition when a local
    torchvision installation has operator-registration issues. In normal training
    environments, torchvision.models.vgg16 is used directly.
    """
    try:
        import torchvision.models as models  # type: ignore

        try:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            vgg = models.vgg16(weights=weights)
        except Exception:
            vgg = models.vgg16(pretrained=pretrained)
        return nn.Sequential(*list(vgg.features.children())[:23])
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to import torchvision VGG16 pretrained features. "
                "Please install a compatible torch/torchvision pair, or set "
                "pretrained_vgg=False for debugging."
            ) from exc

        # Fallback for shape/debug tests only. This follows the VGG16 layout up to
        # conv4_3 but uses randomly initialized weights.
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )


class StructureCompensationBranch(nn.Module):
    """
    Weak-reference structure compensation for DR SR-IQA.

    I_BI   = bicubic(I_LR)
    I_SR^S = G * I_SR
    I_BI^S = G * I_BI
    F_A    = GAP(f(I_SR^S) - f(I_BI^S))
    """

    def __init__(
        self,
        sigma: float = 1.2,
        pretrained_vgg: bool = True,
        normalize_vgg_input: bool = True,
    ):
        super().__init__()
        self.structure_decomposition = GaussianDecomposition(channels=3, sigma=sigma)
        self.vgg = build_vgg16_features(pretrained=pretrained_vgg)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.normalize_vgg_input = normalize_vgg_input
        self.register_buffer("vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_vgg_input:
            return x
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, sr_img: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        bi_img = F.interpolate(lr_img, size=sr_img.shape[2:], mode="bicubic", align_corners=False)

        sr_structure, _ = self.structure_decomposition(sr_img)
        bi_structure, _ = self.structure_decomposition(bi_img)

        sr_structure = self._normalize(sr_structure)
        bi_structure = self._normalize(bi_structure)

        sr_feat = self.vgg(sr_structure)
        bi_feat = self.vgg(bi_structure)

        aux = self.gap(sr_feat - bi_feat).flatten(1)
        return aux


# -----------------------------------------------------------------------------
# Main network: IQANet aligned with Chapter 3 Fig. 3-2
# -----------------------------------------------------------------------------

class TCDINet(nn.Module):
    """
    TCDI-Net: Texture-Structure Complementary Dual-branch Interaction Network.

    NR usage:
        model = TCDINet(dr_mode=False)
        score = model(sr_img)

    DR usage:
        model = TCDINet(dr_mode=True)
        score = model(sr_img, lr_img=lr_img)

    x_t is optional and only kept for ablation/debugging. By default, I_T is
    generated internally from I_SR according to Eq. (3.1)-(3.2).
    """

    def __init__(
        self,
        dr_mode: bool = False,
        in_channels: int = 3,
        sigma: float = 1.2,
        branch_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        gdib_groups: int = 8,
        pretrained_vgg: bool = True,
        normalize_vgg_input: bool = True,
        use_sigmoid: bool = True,
        # Ablation flags (all default True = full model)
        use_tgb: bool = True,
        use_msffb: bool = True,
        use_dcab: bool = True,
        use_bop: bool = True,
        use_gdib: bool = True,
    ):
        super().__init__()
        self.dr_mode = dr_mode
        self.use_sigmoid = use_sigmoid
        c1, c2, c3, c4 = branch_channels

        # Store ablation flags for forward
        self.use_tgb = use_tgb
        self.use_msffb = use_msffb
        self.use_dcab = use_dcab
        self.use_bop = use_bop
        self.use_gdib = use_gdib

        # Eq. (3.1)-(3.3): internal input-level decomposition
        self.decomposition = GaussianDecomposition(channels=in_channels, sigma=sigma)

        # Image branch in Fig. 3-2: [ResBlock + optional DCAB + TGB feedback] x 4
        self.image_stages = nn.ModuleList([
            ImageStage(in_channels, c1, stage_index=0, use_dcab=use_dcab),
            ImageStage(c1, c2, stage_index=1, use_dcab=use_dcab),
            ImageStage(c2, c3, stage_index=2, use_dcab=use_dcab),
            ImageStage(c3, c4, stage_index=3, use_dcab=use_dcab),
        ])

        # TGB modules (only used if use_tgb=True)
        if use_tgb:
            self.tgbs = nn.ModuleList([
                TGB(c1, texture_channels=in_channels),
                TGB(c2, texture_channels=in_channels),
                TGB(c3, texture_channels=in_channels),
                TGB(c4, texture_channels=in_channels),
            ])
        else:
            self.tgbs = None

        # Detail branch for I_T (needed for TGB condition and GDIB)
        self.detail_branch = DetailBranch(in_channels=in_channels, channels=branch_channels)

        # GDIB (only used if use_gdib=True)
        if use_gdib:
            self.gdib = GDIB(det_channels=c4, img_channels=c4, kernel_size=3, groups=gdib_groups)
        else:
            self.gdib = None

        # MSFFB (only used if use_msffb=True)
        if use_msffb:
            self.msffb = MSFFB(list(branch_channels), out_channels=c4)
        else:
            self.msffb = None

        # Determine regressor input dimension based on active modules
        if use_bop and use_gdib:
            nr_dim = 4 * c4       # BOP(fusion) + mean(low) + var(high)
        elif use_bop:
            nr_dim = 2 * c4       # BOP(fusion) only
        else:
            nr_dim = c4           # GAP only

        self.nr_dim = nr_dim

        if dr_mode:
            self.structure_branch = StructureCompensationBranch(
                sigma=sigma,
                pretrained_vgg=pretrained_vgg,
                normalize_vgg_input=normalize_vgg_input,
            )
            dr_dim = 512
            regressor_dim = nr_dim + dr_dim
        else:
            self.structure_branch = None
            regressor_dim = nr_dim

        self.regressor = nn.Sequential(
            nn.Linear(regressor_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        sr_img: torch.Tensor,
        lr_img: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        # Input-level decomposition.
        _, detail_img = self.decomposition(sr_img)
        if x_t is not None:
            detail_img = x_t

        # Detail branch: I_T -> F_T^1...F_T^4
        detail_feats = self.detail_branch(detail_img)

        # Image branch: stage-by-stage processing
        f = sr_img
        img_feats: List[torch.Tensor] = []
        tgb_feats: List[torch.Tensor] = []
        for i, stage in enumerate(self.image_stages):
            stage_feat = stage(f)                     # F_I^i (ResBlock + optional DCAB)
            if self.use_tgb:
                guided_feat = self.tgbs[i](stage_feat, detail_img)  # F_TGB^i
                f = stage_feat + guided_feat           # feedback addition
                tgb_feats.append(guided_feat)
            else:
                f = stage_feat
            img_feats.append(f)

        final_feat = img_feats[-1]                     # F^4

        # GDIB: F^4 + final detail feature -> F_Low, F_High
        dyn_kernel = None
        if self.use_gdib:
            if return_features:
                low, high, dyn_kernel = self.gdib(final_feat, detail_feats[-1], return_kernel=True)
            else:
                low, high = self.gdib(final_feat, detail_feats[-1], return_kernel=False)
        else:
            low = final_feat
            high = None

        # MSFFB: F^1...F^4 -> F_Fusion (or just use F^4 if MSFFB is disabled)
        if self.use_msffb:
            fusion = self.msffb(img_feats)
        else:
            fusion = final_feat

        # Aggregation: BOP or GAP
        if self.use_bop:
            fusion_mean, fusion_var = global_mean_var(fusion)
            if self.use_gdib:
                low_mean, _ = global_mean_var(low)
                _, high_var = global_mean_var(high)
                f_nr = torch.cat([fusion_mean, fusion_var, low_mean, high_var], dim=1)
            else:
                f_nr = torch.cat([fusion_mean, fusion_var], dim=1)
        else:
            # GAP: global average pooling
            gap = F.adaptive_avg_pool2d(low if self.use_gdib else fusion, 1).flatten(1)
            f_nr = gap

        if self.dr_mode:
            if lr_img is None:
                raise ValueError("dr_mode=True requires lr_img for structure compensation.")
            f_aux = self.structure_branch(sr_img, lr_img)
            quality_feat = torch.cat([f_nr, f_aux], dim=1)
        else:
            f_aux = None
            quality_feat = f_nr

        score = self.regressor(quality_feat)
        if self.use_sigmoid:
            score = torch.sigmoid(score)

        if return_features:
            return score, {
                "detail_img": detail_img,
                "detail_feats": detail_feats,
                "img_feats": img_feats,
                "tgb_feats": tgb_feats,
                "final_feat": final_feat,
                "low": low,
                "high": high,
                "fusion": fusion,
                "f_nr": f_nr,
                "f_aux": f_aux,
                "dynamic_kernel": dyn_kernel,
            }
        return score


# -----------------------------------------------------------------------------
# Unified optimization objective: Eq. (3.26)-(3.28)
# -----------------------------------------------------------------------------

class SRIQALoss(nn.Module):
    """
    L = mean(Huber(q_hat, q)) + lambda_plcc * (1 - PLCC_batch)
    """

    def __init__(self, delta: float = 1.0 / 9.0, lambda_plcc: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.delta = delta
        self.lambda_plcc = lambda_plcc
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        err = pred - target
        abs_err = err.abs()
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err.pow(2),
            self.delta * (abs_err - 0.5 * self.delta),
        ).mean()

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        plcc = (pred_centered * target_centered).sum() / (
            torch.sqrt(pred_centered.pow(2).sum() + self.eps)
            * torch.sqrt(target_centered.pow(2).sum() + self.eps)
        )
        plcc_loss = 1.0 - plcc

        return huber + self.lambda_plcc * plcc_loss


# -----------------------------------------------------------------------------
# Ablation study presets (paper Table / Figure)
# -----------------------------------------------------------------------------
# Each key maps to kwargs for TCDINet(…), overriding the defaults.
# "full" is the complete model with ALL modules enabled.

ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "baseline":                {"use_tgb": False, "use_msffb": False, "use_dcab": False, "use_bop": False, "use_gdib": False},
    "tgb":                     {"use_tgb": True,  "use_msffb": False, "use_dcab": False, "use_bop": False, "use_gdib": False},
    "msffb":                   {"use_tgb": False, "use_msffb": True,  "use_dcab": False, "use_bop": False, "use_gdib": False},
    "dcab":                    {"use_tgb": False, "use_msffb": False, "use_dcab": True,  "use_bop": False, "use_gdib": False},
    "bop":                     {"use_tgb": False, "use_msffb": False, "use_dcab": False, "use_bop": True,  "use_gdib": False},
    "gdib":                    {"use_tgb": False, "use_msffb": False, "use_dcab": False, "use_bop": False, "use_gdib": True},
    "tgb+msffb":               {"use_tgb": True,  "use_msffb": True,  "use_dcab": False, "use_bop": False, "use_gdib": False},
    "tgb+msffb+dcab":          {"use_tgb": True,  "use_msffb": True,  "use_dcab": True,  "use_bop": False, "use_gdib": False},
    "tgb+msffb+dcab+bop":      {"use_tgb": True,  "use_msffb": True,  "use_dcab": True,  "use_bop": True,  "use_gdib": False},
    "full":                    {"use_tgb": True,  "use_msffb": True,  "use_dcab": True,  "use_bop": True,  "use_gdib": True},
}


def build_ablation_model(
    variant: str,
    dr_mode: bool = False,
    pretrained_vgg: bool = True,
    **kwargs,
) -> TCDINet:
    """Build a TCDINet for a named ablation variant.

    Args:
        variant: key in ABLATION_CONFIGS (e.g. "baseline", "tgb+msffb", "full").
        dr_mode: enable DR structure compensation branch.
        pretrained_vgg: use pretrained VGG weights in DR branch.

    Returns:
        TCDINet configured for the specified ablation variant.
    """
    if variant not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation variant '{variant}'. Choose from: {list(ABLATION_CONFIGS.keys())}")
    cfg = ABLATION_CONFIGS[variant]
    cfg.update(kwargs)
    return TCDINet(dr_mode=dr_mode, pretrained_vgg=pretrained_vgg, **cfg)


if __name__ == "__main__":
    # Quick shape check. Use pretrained_vgg=False to avoid downloading weights.
    print("Supported ablation variants:", list(ABLATION_CONFIGS.keys()))
    sr = torch.rand(2, 3, 128, 128)
    lr = torch.rand(2, 3, 32, 32)

    # NR full model
    nr_model = TCDINet(dr_mode=False, pretrained_vgg=False)
    nr_score, nr_feats = nr_model(sr, return_features=True)
    print(f"NR full score: {nr_score.shape}, F_NR: {nr_feats['f_nr'].shape}")

    # DR full model
    dr_model = TCDINet(dr_mode=True, pretrained_vgg=False)
    dr_score, dr_feats = dr_model(sr, lr_img=lr, return_features=True)
    print(f"DR full score: {dr_score.shape}, F_aux: {dr_feats['f_aux'].shape}")

    # Ablation: baseline
    base = build_ablation_model("baseline")
    base_score = base(sr)
    print(f"Baseline score: {base_score.shape}, regressor dim: {base.nr_dim}")
