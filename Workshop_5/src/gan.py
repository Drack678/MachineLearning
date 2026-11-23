"""GAN secuencial 1D discreta con Gumbel-Softmax Straight-Through (Workshop 5 — Fase 5)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import WINDOW_SIZE
from shared.data import build_contiguous_blocks, read_vocab

SEQUENCE_LENGTH = WINDOW_SIZE + 1


def load_vocab() -> dict[str, Any]:
    return read_vocab()


class CachedBlockDataset(Dataset):
    def __init__(self, blocks: torch.Tensor) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        return int(self.blocks.size(0))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.blocks[index].long()


def load_training_data(stride: int = 40) -> tuple[CachedBlockDataset, dict[str, Any]]:
    vocab = load_vocab()
    blocks = build_contiguous_blocks("train", vocab=vocab, block_size=SEQUENCE_LENGTH, stride=stride)
    return CachedBlockDataset(blocks), vocab


loadTrainingData = load_training_data


class GenResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        noise_dim: int = 128,
        hidden_dim: int = 256,
        seq_len: int = SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.init_len = 21
        self.hparams = {
            "vocab_size": vocab_size,
            "noise_dim": noise_dim,
            "hidden_dim": hidden_dim,
            "seq_len": seq_len,
        }

        self.fc = nn.Linear(noise_dim, hidden_dim * self.init_len)
        self.stage1 = nn.Sequential(
            GenResBlock1D(hidden_dim, kernel_size=5, dilation=1),
            GenResBlock1D(hidden_dim, kernel_size=5, dilation=2),
        )
        self.up1_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.stage2 = nn.Sequential(
            GenResBlock1D(hidden_dim, kernel_size=5, dilation=1),
            GenResBlock1D(hidden_dim, kernel_size=5, dilation=2),
        )
        self.up2_conv = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.stage3 = nn.Sequential(
            GenResBlock1D(hidden_dim // 2, kernel_size=5, dilation=1),
            GenResBlock1D(hidden_dim // 2, kernel_size=5, dilation=4),
        )
        self.out_conv = nn.Conv1d(hidden_dim // 2, vocab_size, kernel_size=1)

    def forward_logits(self, noise: torch.Tensor) -> torch.Tensor:
        h = self.fc(noise).view(noise.size(0), self.hidden_dim, self.init_len)
        h = self.stage1(h)
        h = F.interpolate(h, size=41, mode="nearest")
        h = F.gelu(self.up1_conv(h))
        h = self.stage2(h)
        h = F.interpolate(h, size=self.seq_len, mode="nearest")
        h = F.gelu(self.up2_conv(h))
        h = self.stage3(h)
        return self.out_conv(h).transpose(1, 2)

    def forward(
        self,
        noise: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
        return_logits: bool = False,
    ) -> torch.Tensor:
        logits = self.forward_logits(noise)
        if return_logits:
            return logits
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class DiscResBlock1D(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 5, dilation: int = 1, use_spectral_norm: bool = True
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        c1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        c2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        if use_spectral_norm:
            c1 = spectral_norm(c1)
            c2 = spectral_norm(c2)
        self.use_layer_norm = not use_spectral_norm
        self.ln1 = nn.LayerNorm(channels) if self.use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(channels) if self.use_layer_norm else nn.Identity()
        self.c1 = c1
        self.c2 = c2

    def _norm(self, x: torch.Tensor, ln: nn.Module) -> torch.Tensor:
        if not self.use_layer_norm:
            return x
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self._norm(x, self.ln1), 0.2)
        h = self.c1(h)
        h = F.leaky_relu(self._norm(h, self.ln2), 0.2)
        h = self.c2(h)
        return x + 0.3 * h


class _SeqDiscriminatorBase(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_dim: int = 192, use_spectral_norm: bool = True
    ) -> None:
        super().__init__()
        self.hparams = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "use_spectral_norm": use_spectral_norm,
        }
        in_conv = nn.Conv1d(vocab_size, hidden_dim, kernel_size=1)
        head = nn.Linear(2 * hidden_dim, 1)
        if use_spectral_norm:
            in_conv = spectral_norm(in_conv)
            head = spectral_norm(head)
        self.in_conv = in_conv
        self.blocks = nn.Sequential(
            DiscResBlock1D(hidden_dim, kernel_size=5, dilation=1, use_spectral_norm=use_spectral_norm),
            DiscResBlock1D(hidden_dim, kernel_size=5, dilation=2, use_spectral_norm=use_spectral_norm),
            DiscResBlock1D(hidden_dim, kernel_size=5, dilation=4, use_spectral_norm=use_spectral_norm),
            DiscResBlock1D(hidden_dim, kernel_size=5, dilation=8, use_spectral_norm=use_spectral_norm),
        )
        self.head = head

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(sequence.transpose(1, 2))
        h = self.blocks(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.head(pooled).squeeze(1)


class Discriminator(_SeqDiscriminatorBase):
    def __init__(self, vocab_size: int, hidden_dim: int = 192) -> None:
        super().__init__(vocab_size=vocab_size, hidden_dim=hidden_dim, use_spectral_norm=True)


class Critic(_SeqDiscriminatorBase):
    def __init__(self, vocab_size: int, hidden_dim: int = 192) -> None:
        super().__init__(vocab_size=vocab_size, hidden_dim=hidden_dim, use_spectral_norm=False)


def gumbel_temperature(epoch: int, total_epochs: int, tau_start: float = 2.0, tau_end: float = 0.5) -> float:
    if total_epochs <= 1:
        return tau_end
    progress = (epoch - 1) / float(total_epochs - 1)
    return float(tau_start * ((tau_end / tau_start) ** progress))
