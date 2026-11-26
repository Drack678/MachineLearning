"""Domain transfer architectures (W2/W4/W5 -> Medical Imaging), training loop, and 4 ablations.

Implements Phase 2 of Workshop 6 with full AMD ROCm RX 9060 XT / CUDA / CPU portability:
- `MedHybridTransformer`: hybrid 2D ConvStem + W2 Pre-LN TransformerEncoderLayer blocks
  (supports `scratch`, `w2_frozen` [honest negative feature-reuse control], and `w2_finetune`).
- `ResNet18Medical`: ImageNet transfer reference (`pretrained` fine-tune / frozen / scratch)
  to benchmark text-to-medical vs natural-image-to-medical transfer (Raghu et al., 2019).
- `MedConvVAE2D`: 1D->2D adaptation of W4 ResConv VAE (GroupNorm+GELU, 2D FiLM, Free Bits,
  Cyclical KL Annealing, L1+SSIM loss) trained on benign lesions for anomaly/OOD scoring.
- `MedCondWGANGP2D`: 1D->2D conditional adaptation of W5 WGAN-GP (FP32 gradient penalty outside
  autocast, LayerNorm/GroupNorm critic, NO Gumbel-ST) for rare-class synthetic augmentation (A4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from shared.device import (  # noqa: E402
    DEVICE_CHOICES,
    PROFILE_CHOICES,
    amp_context,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    get_hardware_profile,
    resolve_device,
    seed_everything,
)
from shared.metrics import apply_logit_adjustment, fit_logit_adjustment_tau  # noqa: E402
from shared.models import build_warmup_cosine_scheduler, count_parameters  # noqa: E402
from Workshop_6.src.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    FIGURES_DIR,
    GAN_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MALIGNANT_CLASS_IDS,
    NUM_CLASSES,
    REPORTS_DIR,
    SEED,
    VAE_IMAGE_SIZE,
    W2_BEST_CHECKPOINT,
    to_relative_path,
)
from Workshop_6.src.medical_metrics import (  # noqa: E402
    TemperatureScaler,
    compute_binary_clinical_metrics,
    compute_multiclass_clinical_metrics,
    malignant_probabilities,
    select_clinical_thresholds_on_val,
)
from Workshop_6.src.medical_preprocessing import (  # noqa: E402
    ClassBalancedFocalLoss,
    HAM10000Dataset,
    build_medical_transforms,
    compute_class_balanced_weights,
    get_cache_dir,
    load_split_dataframes,
)

# ---------------------------------------------------------------------------
# 1. W2 -> W6 Hybrid Architecture: ConvStem2D + Pre-LN Transformer Blocks
# ---------------------------------------------------------------------------


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ConvStem2D(nn.Module):
    """4-stage strided 2D convolutional patch stem mapping (B, 3, H, W) -> (B, d_model, H/16, W/16)."""

    def __init__(self, in_channels: int = 3, d_model: int = 256) -> None:
        super().__init__()
        c1, c2, c3 = 32, 64, 128
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c1), c1),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c2), c2),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c3), c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c3), c3),
            nn.GELU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(d_model), d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.stage4(h)


class MedHybridTransformer(nn.Module):
    """Medical CNN-Transformer adapting W2 CharTransformer Pre-LN blocks to 2D dermoscopy.

    State-dict keys under `self.blocks` match `shared.models.CharTransformer.blocks` 1:1 so
    trained Workshop 2 weights can be transferred cleanly for `w2_frozen` and `w2_finetune`.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        image_size: int = DEFAULT_IMAGE_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}")
        self.grid_size = image_size // 16
        self.num_patches = self.grid_size * self.grid_size
        self.hparams: dict[str, Any] = {
            "arch": "MedHybridTransformer",
            "num_classes": num_classes,
            "image_size": image_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "activation": activation,
        }

        self.stem = ConvStem2D(in_channels=3, d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)

        # Identical Pre-LN TransformerEncoderLayer specification as W2 CharTransformer
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_last_conv_layer(self) -> nn.Module:
        return self.stem.stage4[0]

    def load_w2_blocks(
        self, w2_ckpt_path: Path = W2_BEST_CHECKPOINT, *, freeze_blocks: bool = False
    ) -> dict[str, Any]:
        """Transfer only `blocks.*` weights from W2 CharTransformer; discard text embeddings/head."""
        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        source_found = w2_ckpt_path.exists()

        if source_found:
            sd, _ = load_checkpoint(w2_ckpt_path, device="cpu")
            block_sd = {
                k.removeprefix("blocks."): v for k, v in sd.items() if k.startswith("blocks.")
            }
            skipped_keys = [k for k in sd if not k.startswith("blocks.")]
            incompatible = self.blocks.load_state_dict(block_sd, strict=False)
            loaded_keys = list(block_sd.keys())
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
        else:
            missing = []
            unexpected = []

        if freeze_blocks:
            for p in self.blocks.parameters():
                p.requires_grad = False

        return {
            "w2_checkpoint_found": source_found,
            "w2_checkpoint_path": to_relative_path(w2_ckpt_path),
            "transferred_block_tensors": len(loaded_keys),
            "discarded_text_tensors": len(skipped_keys),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "blocks_frozen": freeze_blocks,
        }

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)  # (B, d_model, Gh, Gw)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N_patches, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        return self.emb_dropout(seq + self.pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._tokens(x)
        h = self.blocks(h)  # Bidirectional self-attention (NO causal mask)
        return self.head(h[:, 0])

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass while capturing per-layer multi-head self-attention weights."""
        h = self._tokens(x)
        attn_maps: list[torch.Tensor] = []
        for layer in self.blocks.layers:
            # Replicate Pre-LN TransformerEncoderLayer forward exactly while requesting attn weights
            h_norm = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                h_norm,
                h_norm,
                h_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_maps.append(attn_w.detach())
            h = h + layer.dropout1(attn_out)
            h_ff = layer.norm2(h)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_ff))))
            h = h + layer.dropout2(ff_out)
        if self.blocks.norm is not None:
            h = self.blocks.norm(h)
        logits = self.head(h[:, 0])
        return logits, attn_maps


# ---------------------------------------------------------------------------
# 2. ImageNet Vision Transfer Reference (Raghu et al., 2019 comparison)
# ---------------------------------------------------------------------------


class ResNet18Medical(nn.Module):
    """ResNet-18 adapted for 7-class dermoscopy (supports ImageNet pretrained or scratch)."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, Any] = {
            "arch": "ResNet18Medical",
            "num_classes": num_classes,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "dropout": dropout,
        }
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load ImageNet weights ({exc}); falling back to scratch.")
            self.backbone = models.resnet18(weights=None)
            self.hparams["pretrained"] = False

        in_features = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def get_last_conv_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_medical_classifier(hparams: dict[str, Any]) -> nn.Module:
    """Factory reconstructing any saved Workshop 6 classifier cleanly from metadata."""
    arch = str(hparams.get("arch", "MedHybridTransformer"))
    if arch == "MedHybridTransformer":
        return MedHybridTransformer(
            num_classes=int(hparams.get("num_classes", NUM_CLASSES)),
            image_size=int(hparams.get("image_size", DEFAULT_IMAGE_SIZE)),
            d_model=int(hparams.get("d_model", 256)),
            n_heads=int(hparams.get("n_heads", 4)),
            n_layers=int(hparams.get("n_layers", 4)),
            dropout=float(hparams.get("dropout", 0.1)),
            activation=str(hparams.get("activation", "gelu")),
        )
    if arch == "ResNet18Medical":
        return ResNet18Medical(
            num_classes=int(hparams.get("num_classes", NUM_CLASSES)),
            pretrained=False,  # Weights will be loaded strictly from checkpoint
            freeze_backbone=False,
            dropout=float(hparams.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown architecture in checkpoint metadata: {arch}")


# ---------------------------------------------------------------------------
# 3. W4 -> W6 Adaptation: 2D Residual Conv VAE with FiLM, Free Bits & Cyclical KL
# ---------------------------------------------------------------------------


class ResConv2dBlock(nn.Module):
    """2D translation of Workshop 4 `ResConv1dBlock` (GroupNorm + GELU + Conv2d)."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        g = _gn_groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MedConvVAE2D(nn.Module):
    """2D Convolutional VAE adapted from W4 for unsupervised benign->malignant anomaly/OOD detection."""

    def __init__(
        self,
        image_size: int = VAE_IMAGE_SIZE,
        z_dim: int = 128,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"VAE image_size must be divisible by 16, got {image_size}")
        self.image_size = image_size
        self.z_dim = z_dim
        self.base_ch = base_ch
        self.feat_hw = image_size // 16
        self.bottleneck_ch = base_ch * 2
        self.hparams = {
            "arch": "MedConvVAE2D",
            "image_size": image_size,
            "z_dim": z_dim,
            "base_ch": base_ch,
        }

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResConv2dBlock(base_ch, dilation=1),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResConv2dBlock(base_ch * 2, dilation=2),
            nn.Conv2d(base_ch * 2, self.bottleneck_ch, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.bottleneck_ch, self.bottleneck_ch, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        flat_dim = self.bottleneck_ch * self.feat_hw * self.feat_hw
        self.enc_fc = nn.Sequential(nn.Linear(flat_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.to_mu = nn.Linear(256, z_dim)
        self.to_logvar = nn.Linear(256, z_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, flat_dim),
            nn.GELU(),
        )
        # 2D FiLM conditioning adapted from W4 `dec_film`
        self.dec_film = nn.Linear(z_dim, 2 * self.bottleneck_ch)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.bottleneck_ch, self.bottleneck_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.bottleneck_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.GELU(),
            ResConv2dBlock(base_ch * 2, dilation=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GELU(),
            ResConv2dBlock(base_ch, dilation=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # continuous [0, 1] pixel range
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        h = self.enc_fc(h)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(z.size(0), self.bottleneck_ch, self.feat_hw, self.feat_hw)
        gamma, beta = self.dec_film(z).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, *, sample_posterior: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        do_sample = self.training if sample_posterior is None else sample_posterior
        z = self.reparameterize(mu, log_var) if do_sample else mu
        return self.decode(z), mu, log_var


def vae_kl_free_bits(
    mu: torch.Tensor, log_var: torch.Tensor, free_bits: float = 0.25
) -> torch.Tensor:
    kl_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0.0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)
    return kl_dim.sum(dim=1)


def vae_cyclical_beta(
    epoch: int, total_epochs: int, target_beta: float = 0.002, n_cycles: int = 4
) -> float:
    cycle_len = max(1.0, total_epochs / float(n_cycles))
    pos = ((epoch - 1) % cycle_len) / cycle_len
    return float(target_beta * (pos / 0.5)) if pos < 0.5 else float(target_beta)


def vae_per_sample_recon_error(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Continuous L1 + MSE reconstruction error per image (replaces W4 discrete CE)."""
    l1 = (recon - target).abs().mean(dim=(1, 2, 3))
    mse = (recon - target).pow(2).mean(dim=(1, 2, 3))
    return l1 + 0.5 * mse


# ---------------------------------------------------------------------------
# 4. W5 -> W6 Adaptation: Conditional 2D WGAN-GP (FP32 GP, NO Gumbel-ST)
# ---------------------------------------------------------------------------


class CondGen2D(nn.Module):
    """Class-conditional 2D Generator adapted from W5 (continuous tanh output, no Gumbel-ST)."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        noise_dim: int = 128,
        class_emb_dim: int = 32,
        base_ch: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.base_ch = base_ch
        self.hparams = {
            "arch": "CondGen2D",
            "num_classes": num_classes,
            "noise_dim": noise_dim,
            "class_emb_dim": class_emb_dim,
            "base_ch": base_ch,
            "image_size": GAN_IMAGE_SIZE,
        }
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.fc = nn.Linear(noise_dim + class_emb_dim, base_ch * 2 * 4 * 4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            ResConv2dBlock(base_ch * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            ResConv2dBlock(base_ch),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            ResConv2dBlock(base_ch // 2),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch // 2, base_ch // 4, kernel_size=3, padding=1),
            nn.GroupNorm(_gn_groups(base_ch // 4), base_ch // 4),
            nn.GELU(),
            nn.Conv2d(base_ch // 4, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.class_emb(labels.long())
        h = self.fc(torch.cat([noise, emb], dim=-1)).view(noise.size(0), self.base_ch * 2, 4, 4)
        h = self.up1(h)  # 8x8
        h = self.up2(h)  # 16x16
        h = self.up3(h)  # 32x32
        return self.up4(h)  # 64x64


class CriticResBlock2D(nn.Module):
    """2D translation of W5 `DiscResBlock1D` with GroupNorm (GP-safe) and 0.3 residual scaling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.gn1(self.c1(x)), 0.2)
        h = F.leaky_relu(self.gn2(self.c2(h)), 0.2)
        return self.skip(x) + 0.3 * h


class CondCritic2D(nn.Module):
    """Projection Conditional Critic 2D for WGAN-GP (strictly NO BatchNorm)."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_ch: int = 64) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            CriticResBlock2D(base_ch, base_ch * 2),  # 32x32
            CriticResBlock2D(base_ch * 2, base_ch * 4),  # 16x16
            CriticResBlock2D(base_ch * 4, base_ch * 4),  # 8x8
            CriticResBlock2D(base_ch * 4, base_ch * 4),  # 4x4
        )
        feat_dim = base_ch * 4
        self.head = nn.Linear(feat_dim, 1)
        self.class_proj = nn.Embedding(num_classes, feat_dim)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.in_conv(images), 0.2)
        h = self.blocks(h)
        pooled = h.mean(dim=(2, 3))
        uncond = self.head(pooled).squeeze(1)
        cond = (pooled * self.class_proj(labels.long())).sum(dim=1)
        return uncond + cond


def gradient_penalty_2d(
    critic: CondCritic2D,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """2D WGAN-GP gradient penalty forced in FP32 outside autocast (safe on ROCm RX 9060 XT)."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        real_f = real.float()
        fake_f = fake.detach().float()
        bs = real_f.size(0)
        alpha = torch.rand(bs, 1, 1, 1, device=device, dtype=torch.float32)
        interpolated = (alpha * real_f + (1.0 - alpha) * fake_f).requires_grad_(True)
        scores = critic(interpolated, labels)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.reshape(bs, -1)
        grad_norm = torch.sqrt(gradients.pow(2).sum(dim=1) + 1e-12)
        return ((grad_norm - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# 5. Reproducible Classifier Training Loop (AMP, Early Stopping, Val Calibration)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_classifier_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run deterministic inference returning raw CPU float32 logits and true labels."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[np.ndarray] = []
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xb)
        all_logits.append(logits.float().cpu())
        all_targets.append(yb.numpy() if isinstance(yb, torch.Tensor) else np.asarray(yb))
    return torch.cat(all_logits, dim=0), np.concatenate(all_targets, axis=0)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    use_foreach: bool | None,
    differential_blocks_lr_scale: float | None = None,
) -> torch.optim.Optimizer:
    """Create AdamW with optional differential LR for transferred W2 Transformer blocks."""
    if differential_blocks_lr_scale is not None and hasattr(model, "blocks"):
        block_params = [p for p in model.blocks.parameters() if p.requires_grad]
        block_ids = {id(p) for p in block_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in block_ids]
        groups: list[dict[str, Any]] = []
        if other_params:
            groups.append({"params": other_params, "lr": lr})
        if block_params:
            groups.append({"params": block_params, "lr": lr * differential_blocks_lr_scale})
        return torch.optim.AdamW(
            groups, betas=(0.9, 0.95), weight_decay=weight_decay, foreach=use_foreach
        )
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, foreach=use_foreach
    )


def train_medical_classifier(
    *,
    run_name: str,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_class_counts: np.ndarray,
    device: torch.device,
    profile_name: str = "auto",
    epochs: int = 25,
    batch_size: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 6,
    balance_mode: str = "cb_focal",  # "none" | "cb_focal" | "logit_adj"
    differential_lr_scale: float | None = None,
    seed: int = SEED,
    num_workers: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
    save_ckpt: bool = True,
) -> dict[str, Any]:
    """Train a Workshop 6 medical classifier with AMP, early stopping, and val-only calibration."""
    seed_everything(seed, deterministic=True)
    profile = get_hardware_profile(device, profile_name)
    # Image models use smaller batch size than 1D text sequences
    default_bs = 64 if profile.name in ("rocm-rx9060", "high-vram") else 32
    bs = int(batch_size or default_bs)

    model = model.to(device)
    dl_kw = dataloader_kwargs(device, num_workers=num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        **dl_kw,
    )
    val_loader = DataLoader(val_dataset, batch_size=bs * 2, shuffle=False, **dl_kw)

    if balance_mode == "cb_focal":
        weights = compute_class_balanced_weights(train_class_counts, beta=0.999).to(device)
        criterion: nn.Module = ClassBalancedFocalLoss(
            class_weights=weights, gamma=2.0, label_smoothing=0.05
        )
    elif balance_mode in ("none", "logit_adj"):
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown balance_mode: {balance_mode}")

    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        use_foreach=profile.use_foreach,
        differential_blocks_lr_scale=differential_lr_scale,
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=0.08, min_lr_ratio=0.01
    )
    scaler = torch.amp.GradScaler("cuda") if profile.use_grad_scaler else None

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    t_start = time.time()

    print(
        f"[{run_name}] params={count_parameters(model):,} | train={len(train_dataset):,} "
        f"| val={len(val_dataset):,} | balance={balance_mode} | device={device} ({profile.name}) | bs={bs}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_seen = 0
        skipped = 0

        for batch in train_loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda")).long()
            optimizer.zero_grad(set_to_none=True)

            with amp_context(device, enabled=profile.amp_enabled, dtype=profile.amp_dtype):
                logits = model(xb)
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if not is_fin:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    skipped += 1
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if not is_fin:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                optimizer.step()

            scheduler.step()
            loss_sum += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        val_logits, val_y = predict_classifier_logits(
            model, val_loader, device, amp_enabled=profile.amp_enabled, amp_dtype=profile.amp_dtype
        )
        val_probs = torch.softmax(val_logits, dim=-1).numpy()
        val_m = compute_multiclass_clinical_metrics(val_y, val_probs)
        mal_m = val_m["malignant_default"]
        # Clinical selection criterion: balances rare-disease multiclass recall & malignant PR-AUC
        sel_score = 0.5 * float(val_m["balanced_accuracy"]) + 0.5 * float(
            mal_m["pr_auc"] if np.isfinite(mal_m["pr_auc"]) else val_m["balanced_accuracy"]
        )

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "val_auc_roc_macro": val_m["auc_roc_ovr_macro"],
            "val_malignant_auc": mal_m["auc_roc"],
            "val_malignant_pr_auc": mal_m["pr_auc"],
            "val_malignant_sens": mal_m["sensitivity"],
            "val_ece": val_m["ece"],
            "select_score": sel_score,
            "skipped_batches": skipped,
        }
        history.append(row)

        improved = sel_score > best_score + 1e-4
        print(
            f"  ep {epoch:02d}/{epochs} | loss {row['train_loss']:.4f} "
            f"| val bal_acc {row['val_balanced_accuracy']:.4f} "
            f"| mal AUC {row['val_malignant_auc']:.4f} | mal PR-AUC {row['val_malignant_pr_auc']:.4f} "
            f"| ECE {row['val_ece']:.4f} ({time.time() - t0:.1f}s){' *best*' if improved else ''}"
        )
        if improved:
            best_score = sel_score
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is None:
        raise RuntimeError(f"[{run_name}] Failed to produce any valid checkpoint state.")

    # Post-hoc validation-only calibration (Temperature Scaling + optional Logit Adjustment + Thresholds)
    model.load_state_dict(best_state)
    best_val_logits, val_y = predict_classifier_logits(
        model, val_loader, device, amp_enabled=profile.amp_enabled, amp_dtype=profile.amp_dtype
    )

    tau_opt = 0.0
    logits_for_cal = best_val_logits
    if balance_mode == "logit_adj":
        tau_opt, _ = fit_logit_adjustment_tau(best_val_logits, val_y, train_class_counts)
        logits_for_cal = apply_logit_adjustment(best_val_logits, train_class_counts, tau_opt)

    temp_scaler = TemperatureScaler()
    temperature = temp_scaler.fit(logits_for_cal, torch.from_numpy(val_y))
    cal_val_probs = torch.softmax(logits_for_cal / temperature, dim=-1).numpy()

    val_metrics_final = compute_multiclass_clinical_metrics(val_y, cal_val_probs)
    y_val_mal = np.isin(val_y, list(MALIGNANT_CLASS_IDS)).astype(np.int64)
    p_val_mal = malignant_probabilities(cal_val_probs)
    clinical_thresholds = select_clinical_thresholds_on_val(y_val_mal, p_val_mal)

    train_seconds = time.time() - t_start
    metadata: dict[str, Any] = {
        "run_name": run_name,
        "model_hparams": getattr(model, "hparams", {}),
        "trainable_params": count_parameters(model),
        "train_hparams": {
            "epochs": epochs,
            "best_epoch": best_epoch,
            "batch_size": bs,
            "lr": lr,
            "weight_decay": weight_decay,
            "balance_mode": balance_mode,
            "differential_lr_scale": differential_lr_scale,
            "seed": seed,
            "profile": profile.name,
        },
        "calibration": {
            "temperature": temperature,
            "logit_adj_tau": tau_opt,
            "train_class_counts": train_class_counts.tolist(),
        },
        "clinical_thresholds_val": clinical_thresholds,
        "val_summary": {
            "balanced_accuracy": val_metrics_final["balanced_accuracy"],
            "f1_macro": val_metrics_final["f1_macro"],
            "auc_roc_ovr_macro": val_metrics_final["auc_roc_ovr_macro"],
            "pr_auc_macro": val_metrics_final["pr_auc_macro"],
            "ece": val_metrics_final["ece"],
            "malignant_auc": val_metrics_final["malignant_default"]["auc_roc"],
            "malignant_pr_auc": val_metrics_final["malignant_default"]["pr_auc"],
        },
        "train_seconds": train_seconds,
        "history": history,
        **(extra_metadata or {}),
    }

    ckpt_path = CHECKPOINTS_DIR / f"{run_name}.pt"
    if save_ckpt:
        save_checkpoint(ckpt_path, best_state, metadata=metadata)
        print(
            f"  [Saved] {to_relative_path(ckpt_path)} | T={temperature:.3f}, tau={tau_opt:.2f} "
            f"| val bal_acc={val_metrics_final['balanced_accuracy']:.4f}, "
            f"mal_AUC={val_metrics_final['malignant_default']['auc_roc']:.4f}"
        )

    return {"metadata": metadata, "checkpoint": to_relative_path(ckpt_path)}


# ---------------------------------------------------------------------------
# 6. W4 Benign VAE Anomaly Detector & W5 Conditional WGAN-GP Synthetic Generator
# ---------------------------------------------------------------------------


def train_benign_vae_anomaly(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    profile_name: str = "auto",
    seed: int = SEED,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Train 2D Conv VAE strictly on benign lesions (nv, bkl) for unsupervised anomaly detection."""
    seed_everything(seed, deterministic=True)
    profile = get_hardware_profile(device, profile_name)
    benign_df = train_df[train_df["is_malignant"] == 0].reset_index(drop=True)

    tf = transforms.Compose(
        [transforms.Resize((VAE_IMAGE_SIZE, VAE_IMAGE_SIZE)), transforms.ToTensor()]
    )
    cache_dir = get_cache_dir(DEFAULT_IMAGE_SIZE, enable_medical_preproc=True)
    train_ds = HAM10000Dataset(
        benign_df,
        image_size=VAE_IMAGE_SIZE,
        enable_medical_preproc=True,
        transform=tf,
        cache_dir=cache_dir if cache_dir.exists() else None,
    )
    val_ds = HAM10000Dataset(
        val_df,
        image_size=VAE_IMAGE_SIZE,
        enable_medical_preproc=True,
        transform=tf,
        cache_dir=cache_dir if cache_dir.exists() else None,
    )
    dl_kw = dataloader_kwargs(device, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **dl_kw)

    model = MedConvVAE2D(image_size=VAE_IMAGE_SIZE, z_dim=128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-2, foreach=profile.use_foreach
    )
    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    print(f"[w4_med_conv_vae2d] Training on {len(benign_df):,} benign lesions ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        beta_now = vae_cyclical_beta(epoch, epochs, target_beta=0.002, n_cycles=4)
        loss_sum = 0.0
        n_seen = 0
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            recon, mu, log_var = model(xb, sample_posterior=True)
            rec = vae_per_sample_recon_error(recon, xb).mean()
            kl = vae_kl_free_bits(mu, log_var, free_bits=0.25).mean()
            loss = rec + beta_now * kl
            if torch.isfinite(loss):
                loss.backward()
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if is_fin:
                    optimizer.step()
                loss_sum += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)

        val_scores, val_y_mal = score_vae_anomalies(model, val_loader, device)
        val_m = compute_binary_clinical_metrics(val_y_mal, val_scores / max(val_scores.max(), 1e-6))
        auc = float(val_m["auc_roc"]) if np.isfinite(val_m["auc_roc"]) else 0.5
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "val_auc": auc})
        if auc >= best_auc or best_state is None:
            best_auc = auc
            best_state = cpu_state_dict(model)

    assert best_state is not None
    model.load_state_dict(best_state)
    val_scores, val_y_mal = score_vae_anomalies(model, val_loader, device)
    # Normalize anomaly scores to [0, 1] using validation quantiles for threshold selection
    s_min = float(np.quantile(val_scores, 0.01))
    s_max = float(np.quantile(val_scores, 0.99))
    val_norm = np.clip((val_scores - s_min) / max(s_max - s_min, 1e-6), 0.0, 1.0)
    thresholds = select_clinical_thresholds_on_val(val_y_mal, val_norm)

    meta = {
        "run_name": "w4_med_conv_vae2d",
        "model_hparams": model.hparams,
        "trainable_params": count_parameters(model),
        "anomaly_norm_bounds": {"q01": s_min, "q99": s_max},
        "clinical_thresholds_val": thresholds,
        "val_anomaly_auc": best_auc,
        "history": history,
    }
    ckpt_path = CHECKPOINTS_DIR / "w4_med_conv_vae2d.pt"
    save_checkpoint(ckpt_path, best_state, metadata=meta)
    print(f"  [Saved] {to_relative_path(ckpt_path)} | val malignant anomaly AUC={best_auc:.4f}")
    return {"metadata": meta, "checkpoint": to_relative_path(ckpt_path)}


@torch.inference_mode()
def score_vae_anomalies(
    model: MedConvVAE2D, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-image anomaly scores (recon + 0.001*KL) and binary malignant ground truth."""
    model.eval()
    scores: list[np.ndarray] = []
    y_mal: list[np.ndarray] = []
    for batch in loader:
        xb, yb = batch[0].to(device), batch[1]
        recon, mu, log_var = model(xb, sample_posterior=False)
        rec = vae_per_sample_recon_error(recon, xb)
        kl = vae_kl_free_bits(mu, log_var, free_bits=0.0)
        sc = (rec + 0.001 * kl).cpu().numpy()
        scores.append(sc)
        y_np = yb.numpy() if isinstance(yb, torch.Tensor) else np.asarray(yb)
        y_mal.append(np.isin(y_np, list(MALIGNANT_CLASS_IDS)).astype(np.int64))
    return np.concatenate(scores, axis=0), np.concatenate(y_mal, axis=0)


def train_conditional_wgan_gp_2d(
    train_df: pd.DataFrame,
    device: torch.device,
    *,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-4,
    critic_steps: int = 5,
    lambda_gp: float = 10.0,
    seed: int = SEED,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Train class-conditional 2D WGAN-GP (W5 adaptation) on HAM10000 training images."""
    seed_everything(seed, deterministic=True)
    tf = transforms.Compose(
        [
            transforms.Resize((GAN_IMAGE_SIZE, GAN_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    cache_dir = get_cache_dir(DEFAULT_IMAGE_SIZE, enable_medical_preproc=True)
    ds = HAM10000Dataset(
        train_df,
        image_size=GAN_IMAGE_SIZE,
        enable_medical_preproc=True,
        transform=tf,
        cache_dir=cache_dir if cache_dir.exists() else None,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **dataloader_kwargs(device, num_workers=num_workers),
    )

    gen = CondGen2D(num_classes=NUM_CLASSES, noise_dim=128).to(device)
    critic = CondCritic2D(num_classes=NUM_CLASSES).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    history: list[dict[str, float]] = []
    print(f"[w5_cond_wgan_gp_2d] Training 2D cWGAN-GP ({epochs} epochs, 64x64)...")
    for epoch in range(1, epochs + 1):
        gen.train()
        critic.train()
        c_sum, g_sum, c_steps, g_steps = 0.0, 0.0, 0, 0
        for step_idx, (real, labels) in enumerate(loader, start=1):
            real = real.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda")).long()
            bs = real.size(0)

            opt_c.zero_grad(set_to_none=True)
            z = torch.randn(bs, gen.noise_dim, device=device)
            with torch.no_grad():
                fake = gen(z, labels)
            gp = gradient_penalty_2d(critic, real, fake, labels, device)
            c_loss = critic(fake, labels).mean() - critic(real, labels).mean() + lambda_gp * gp
            if torch.isfinite(c_loss):
                c_loss.backward()
                _, fin_c = clip_grad_norm_safe(critic.parameters(), max_norm=10.0)
                if fin_c:
                    opt_c.step()
                c_sum += float(c_loss.item())
                c_steps += 1

            if step_idx % critic_steps == 0:
                opt_g.zero_grad(set_to_none=True)
                z = torch.randn(bs, gen.noise_dim, device=device)
                gen_imgs = gen(z, labels)
                g_loss = -critic(gen_imgs, labels).mean()
                if torch.isfinite(g_loss):
                    g_loss.backward()
                    _, fin_g = clip_grad_norm_safe(gen.parameters(), max_norm=5.0)
                    if fin_g:
                        opt_g.step()
                    g_sum += float(g_loss.item())
                    g_steps += 1

        history.append(
            {
                "epoch": epoch,
                "critic_loss": c_sum / max(c_steps, 1),
                "generator_loss": g_sum / max(g_steps, 1) if g_steps else float("nan"),
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  WGAN ep {epoch:02d}/{epochs} | C {history[-1]['critic_loss']:.4f} "
                f"| G {history[-1]['generator_loss']:.4f}"
            )

    _save_gan_sample_grid(gen, device, FIGURES_DIR / "wgan_gp_2d_synthetic_samples.png")
    meta = {
        "run_name": "w5_cond_wgan_gp_2d",
        "generator_hparams": gen.hparams,
        "trainable_params": count_parameters(gen),
        "history": history,
    }
    ckpt_path = CHECKPOINTS_DIR / "w5_cond_wgan_gp_2d.pt"
    save_checkpoint(ckpt_path, cpu_state_dict(gen), metadata=meta)
    print(f"  [Saved] {to_relative_path(ckpt_path)}")
    return {"metadata": meta, "checkpoint": to_relative_path(ckpt_path)}


@torch.inference_mode()
def _save_gan_sample_grid(gen: CondGen2D, device: torch.device, out_path: Path) -> None:
    gen.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(8, 12))
    for c_idx, cname in enumerate(CLASS_NAMES):
        z = torch.randn(4, gen.noise_dim, device=device)
        lbl = torch.full((4,), c_idx, dtype=torch.long, device=device)
        imgs = (gen(z, lbl).cpu().clamp(-1, 1) + 1.0) * 0.5
        for j in range(4):
            axes[c_idx, j].imshow(imgs[j].permute(1, 2, 0).numpy())
            axes[c_idx, j].axis("off")
            if j == 0:
                axes[c_idx, j].set_title(f"{cname.upper()}", fontsize=9, fontweight="bold")
    fig.suptitle("Conditional 2D WGAN-GP Synthetic Dermoscopy Samples (64x64)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


class SyntheticImageDataset(Dataset):
    """Dataset of pre-normalized synthetic images returning `(Tensor, int)` to match `HAM10000Dataset`."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor | list[int]) -> None:
        self.images = images.detach().cpu().float()
        if isinstance(labels, torch.Tensor):
            self.labels = [int(v) for v in labels.detach().cpu().tolist()]
        else:
            self.labels = [int(v) for v in labels]
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have matching length")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


@torch.inference_mode()
def build_filtered_synthetic_dataset(
    gen: CondGen2D,
    filter_classifier: nn.Module,
    *,
    minority_class_ids: tuple[int, ...] = (0, 1, 3, 4, 6),  # akiec, bcc, df, mel, vasc
    samples_per_class: int = 200,
    oversample_factor: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: list[float] | tuple[float, float, float] = IMAGENET_MEAN,
    std: list[float] | tuple[float, float, float] = IMAGENET_STD,
    device: torch.device | None = None,
    seed: int = SEED,
) -> SyntheticImageDataset:
    """Generate rare-class images with W5 cWGAN-GP and keep top-50% by classifier confidence."""
    seed_everything(seed, deterministic=True)
    dev = device if device is not None else torch.device("cpu")
    gen.eval()
    filter_classifier.eval()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    kept_imgs: list[torch.Tensor] = []
    kept_lbls: list[torch.Tensor] = []

    for cid in minority_class_ids:
        n_gen = samples_per_class * oversample_factor
        z = torch.randn(n_gen, gen.noise_dim, device=dev)
        lbl = torch.full((n_gen,), cid, dtype=torch.long, device=dev)
        raw_imgs = (gen(z, lbl).clamp(-1.0, 1.0) + 1.0) * 0.5  # [0, 1] 64x64
        resized = F.interpolate(
            raw_imgs, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        normed = (resized.cpu() - mean_t) / std_t

        confs: list[torch.Tensor] = []
        for s in range(0, len(normed), 64):
            logits = filter_classifier(normed[s : s + 64].to(dev))
            probs = torch.softmax(logits.float(), dim=-1)[:, cid].cpu()
            confs.append(probs)
        scores = torch.cat(confs, dim=0)
        top_idx = scores.argsort(descending=True)[:samples_per_class]
        kept_imgs.append(normed[top_idx])
        kept_lbls.append(torch.full((len(top_idx),), cid, dtype=torch.long))

    return SyntheticImageDataset(torch.cat(kept_imgs, dim=0), torch.cat(kept_lbls, dim=0))


# ---------------------------------------------------------------------------
# 7. Orchestrator for All 4 Mandatory Ablations + Fast-Verify Smoke Mode
# ---------------------------------------------------------------------------


def _make_synthetic_fast_verify_data(
    n_train: int = 84, n_val: int = 28, image_size: int = 64, seed: int = SEED
) -> tuple[TensorDataset, TensorDataset, np.ndarray]:
    """Create tiny deterministic synthetic dataset for CI/fast-verify end-to-end checks."""
    rng = np.random.default_rng(seed)
    x_tr = torch.from_numpy(
        rng.standard_normal((n_train, 3, image_size, image_size)).astype(np.float32)
    )
    y_tr = torch.tensor([i % NUM_CLASSES for i in range(n_train)], dtype=torch.long)
    x_va = torch.from_numpy(
        rng.standard_normal((n_val, 3, image_size, image_size)).astype(np.float32)
    )
    y_va = torch.tensor([i % NUM_CLASSES for i in range(n_val)], dtype=torch.long)
    counts = np.bincount(y_tr.numpy(), minlength=NUM_CLASSES)
    return TensorDataset(x_tr, y_tr), TensorDataset(x_va, y_va), counts


def run_ablation_suite(args: argparse.Namespace) -> None:
    seed_everything(args.seed, deterministic=True)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.fast_verify:
        img_sz = 64
        tr_ds, va_ds, counts = _make_synthetic_fast_verify_data(image_size=img_sz, seed=args.seed)
        model = MedHybridTransformer(num_classes=NUM_CLASSES, image_size=img_sz)
        res = train_medical_classifier(
            run_name="smoke_med_hybrid",
            model=model,
            train_dataset=tr_ds,
            val_dataset=va_ds,
            train_class_counts=counts,
            device=device,
            profile_name=args.profile,
            epochs=2,
            batch_size=16,
            lr=1e-3,
            balance_mode="cb_focal",
            seed=args.seed,
            save_ckpt=False,
        )
        print("[FAST-VERIFY OK]", json.dumps(res["metadata"]["val_summary"], indent=2))
        return

    dfs, split_info = load_split_dataframes()
    train_df, val_df = dfs["train"], dfs["val"]
    train_counts = np.bincount(train_df["label"].to_numpy(), minlength=NUM_CLASSES)

    stats_pre = split_info["train_stats"]["medical_preproc"]
    stats_raw = split_info["train_stats"]["raw"]

    def make_datasets(enable_preproc: bool) -> tuple[HAM10000Dataset, HAM10000Dataset]:
        st = stats_pre if enable_preproc else stats_raw
        cache = get_cache_dir(args.image_size, enable_medical_preproc=enable_preproc)
        tr_tf = build_medical_transforms(
            args.image_size, is_train=True, mean=st["mean"], std=st["std"]
        )
        va_tf = build_medical_transforms(
            args.image_size, is_train=False, mean=st["mean"], std=st["std"]
        )
        tr_ds = HAM10000Dataset(
            train_df,
            image_size=args.image_size,
            enable_medical_preproc=enable_preproc,
            transform=tr_tf,
            cache_dir=cache if cache.exists() else None,
        )
        va_ds = HAM10000Dataset(
            val_df,
            image_size=args.image_size,
            enable_medical_preproc=enable_preproc,
            transform=va_tf,
            cache_dir=cache if cache.exists() else None,
        )
        return tr_ds, va_ds

    tr_pre, va_pre = make_datasets(enable_preproc=True)
    tr_raw, va_raw = make_datasets(enable_preproc=False)
    summary_csv_path = REPORTS_DIR / "ablation_val_summary.csv"
    summary_rows: list[dict[str, Any]] = (
        pd.read_csv(summary_csv_path).to_dict(orient="records")
        if summary_csv_path.exists()
        else []
    )

    def record(ablation_id: str, label: str, res: dict[str, Any]) -> None:
        m = res["metadata"]
        vs = m["val_summary"]
        new_row = {
            "ablation": ablation_id,
            "run_name": m["run_name"],
            "description": label,
            "params": m["trainable_params"],
            "best_epoch": m["train_hparams"]["best_epoch"],
            "val_balanced_accuracy": vs["balanced_accuracy"],
            "val_f1_macro": vs["f1_macro"],
            "val_auc_roc_ovr_macro": vs["auc_roc_ovr_macro"],
            "val_malignant_auc": vs["malignant_auc"],
            "val_malignant_pr_auc": vs["malignant_pr_auc"],
            "val_ece": vs["ece"],
            "train_minutes": m["train_seconds"] / 60.0,
            "checkpoint": res["checkpoint"],
        }
        summary_rows[:] = [r for r in summary_rows if r.get("run_name") != m["run_name"]]
        summary_rows.append(new_row)
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    w2_ckpt = Path(args.w2_ckpt) if args.w2_ckpt else W2_BEST_CHECKPOINT
    stage = args.run

    # --- Ablation 1: Domain Transfer Strategy ---
    if stage in ("all", "a1"):
        m_scratch = MedHybridTransformer(num_classes=NUM_CLASSES, image_size=args.image_size)
        record(
            "A1_transfer",
            "MedHybridTransformer (W2 architecture from scratch)",
            train_medical_classifier(
                run_name="a1_med_hybrid_scratch",
                model=m_scratch,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=3e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
            ),
        )

        m_w2_frozen = MedHybridTransformer(num_classes=NUM_CLASSES, image_size=args.image_size)
        t_info_f = m_w2_frozen.load_w2_blocks(w2_ckpt, freeze_blocks=True)
        record(
            "A1_transfer",
            "MedHybridTransformer (W2 blocks frozen — negative control)",
            train_medical_classifier(
                run_name="a1_med_hybrid_w2_frozen",
                model=m_w2_frozen,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=5e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
                extra_metadata={"w2_transfer_audit": t_info_f},
            ),
        )

        m_w2_ft = MedHybridTransformer(num_classes=NUM_CLASSES, image_size=args.image_size)
        t_info_ft = m_w2_ft.load_w2_blocks(w2_ckpt, freeze_blocks=False)
        record(
            "A1_transfer",
            "MedHybridTransformer (W2 blocks fine-tuned, diff LR 0.2x)",
            train_medical_classifier(
                run_name="a1_med_hybrid_w2_finetune",
                model=m_w2_ft,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=3e-4,
                differential_lr_scale=0.2,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
                extra_metadata={"w2_transfer_audit": t_info_ft},
            ),
        )

        m_rn_imnet = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=True)
        record(
            "A1_transfer",
            "ResNet-18 (ImageNet pretrained fine-tune — visual transfer benchmark)",
            train_medical_classifier(
                run_name="a1_resnet18_imagenet_finetune",
                model=m_rn_imnet,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=2e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
            ),
        )

        m_rn_scratch = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=False)
        record(
            "A1_transfer",
            "ResNet-18 (from scratch — Raghu et al. baseline)",
            train_medical_classifier(
                run_name="a1_resnet18_scratch",
                model=m_rn_scratch,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=5e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
            ),
        )

    # --- Ablation 2: With vs Without Modality-Specific Dermoscopy Preprocessing ---
    if stage in ("all", "a2"):
        m_no_pre = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=True)
        record(
            "A2_preprocessing",
            "ResNet-18 ImageNet WITHOUT dermoscopy preprocessing (raw resize)",
            train_medical_classifier(
                run_name="a2_resnet18_no_med_preproc",
                model=m_no_pre,
                train_dataset=tr_raw,
                val_dataset=va_raw,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=2e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
                extra_metadata={"enable_medical_preproc": False},
            ),
        )

        m_hyb_no_pre = MedHybridTransformer(num_classes=NUM_CLASSES, image_size=args.image_size)
        record(
            "A2_preprocessing",
            "MedHybridTransformer WITHOUT dermoscopy preprocessing (raw resize)",
            train_medical_classifier(
                run_name="a2_med_hybrid_no_med_preproc",
                model=m_hyb_no_pre,
                train_dataset=tr_raw,
                val_dataset=va_raw,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=3e-4,
                balance_mode="cb_focal",
                seed=args.seed,
                num_workers=args.num_workers,
                extra_metadata={"enable_medical_preproc": False},
            ),
        )

    # --- Ablation 3: Class Imbalance Mitigation (None vs CB-Focal vs W2 Logit Adjustment) ---
    if stage in ("all", "a3"):
        m_nobal = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=True)
        record(
            "A3_class_balance",
            "ResNet-18 ImageNet WITHOUT class balancing (standard CE)",
            train_medical_classifier(
                run_name="a3_resnet18_no_balance_ce",
                model=m_nobal,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=2e-4,
                balance_mode="none",
                seed=args.seed,
                num_workers=args.num_workers,
            ),
        )

        m_logit_adj = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=True)
        record(
            "A3_class_balance",
            "ResNet-18 ImageNet + W2 Post-hoc Logit Adjustment (calibrated CE)",
            train_medical_classifier(
                run_name="a3_resnet18_w2_logit_adj",
                model=m_logit_adj,
                train_dataset=tr_pre,
                val_dataset=va_pre,
                train_class_counts=train_counts,
                device=device,
                profile_name=args.profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=2e-4,
                balance_mode="logit_adj",
                seed=args.seed,
                num_workers=args.num_workers,
            ),
        )

    # --- W4 Adapted 2D Conv VAE Anomaly Detector ---
    if stage in ("all", "vae"):
        train_benign_vae_anomaly(
            train_df,
            val_df,
            device,
            epochs=args.vae_epochs,
            batch_size=args.batch_size or 64,
            profile_name=args.profile,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    # --- W5 Adapted 2D Conditional WGAN-GP + Ablation 4 Synthetic Augmentation ---
    if stage in ("all", "wgan", "a4"):
        wgan_ckpt = CHECKPOINTS_DIR / "w5_cond_wgan_gp_2d.pt"
        if stage in ("all", "wgan") or not wgan_ckpt.exists():
            train_conditional_wgan_gp_2d(
                train_df,
                device,
                epochs=args.gan_epochs,
                batch_size=args.batch_size or 64,
                seed=args.seed,
                num_workers=args.num_workers,
            )

        if stage in ("all", "a4"):
            gen_sd, gen_meta = load_checkpoint(wgan_ckpt, device="cpu")
            gen = CondGen2D(
                num_classes=int(gen_meta["generator_hparams"]["num_classes"]),
                noise_dim=int(gen_meta["generator_hparams"]["noise_dim"]),
            ).to(device)
            gen.load_state_dict(gen_sd)

            # Filter synthetic samples using trained A1 ResNet-18 (or hybrid fallback)
            filter_ckpt = CHECKPOINTS_DIR / "a1_resnet18_imagenet_finetune.pt"
            if not filter_ckpt.exists():
                filter_ckpt = CHECKPOINTS_DIR / "a1_med_hybrid_scratch.pt"
            flt_sd, flt_meta = load_checkpoint(filter_ckpt, device="cpu")
            flt_model = build_medical_classifier(flt_meta["model_hparams"]).to(device)
            flt_model.load_state_dict(flt_sd)

            syn_ds = build_filtered_synthetic_dataset(
                gen,
                flt_model,
                samples_per_class=200,
                image_size=args.image_size,
                mean=stats_pre["mean"],
                std=stats_pre["std"],
                device=device,
                seed=args.seed,
            )
            combined_train = ConcatDataset([tr_pre, syn_ds])
            m_a4 = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=True)
            record(
                "A4_wgan_aug",
                "ResNet-18 ImageNet + W5 cWGAN-GP 2D filtered rare-class augmentation",
                train_medical_classifier(
                    run_name="a4_resnet18_wgan_gp_aug",
                    model=m_a4,
                    train_dataset=combined_train,
                    val_dataset=va_pre,
                    train_class_counts=train_counts,
                    device=device,
                    profile_name=args.profile,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=2e-4,
                    balance_mode="cb_focal",
                    seed=args.seed,
                    num_workers=args.num_workers,
                    extra_metadata={"synthetic_samples_added": len(syn_ds)},
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workshop 6 Domain Transfer & Ablation Runner (ROCm RX 9060 XT / CUDA / CPU)."
    )
    parser.add_argument(
        "--run",
        default="all",
        choices=("all", "a1", "a2", "a3", "a4", "vae", "wgan"),
        help="Which ablation/model group to train.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--gan-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--w2-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES)
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    run_ablation_suite(args)


if __name__ == "__main__":
    main()
