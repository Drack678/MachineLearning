"""Autoencoder (AE) y Variational Autoencoder (VAE) secuenciales 1D (Workshop 4 — Fase 4)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint
from shared.config import SEED, W2_BEST_CHECKPOINT, WINDOW_SIZE
from shared.data import load_encoded_split, read_vocab
from shared.device import resolve_device, seed_everything
from shared.models import CharTransformer

WINDOW = WINDOW_SIZE
CHECKPOINTS_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "workshop4_v2"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

N_SAMPLES = {"train": 400_000, "val": 40_000, "test": 40_000}


def set_seeds(seed: int = SEED) -> None:
    seed_everything(seed, deterministic=True)


def get_device(preferred: str = "auto") -> torch.device:
    device, _ = resolve_device(preferred)
    return device


def load_split(split: str, vocab: dict[str, Any] | None = None, max_samples: int | None = None):
    x, y = load_encoded_split(split, vocab=vocab)  # type: ignore[arg-type]
    limit = max_samples if max_samples is not None else N_SAMPLES.get(split)
    if limit is not None and limit < len(x):
        idx = np.sort(np.random.default_rng(SEED).choice(len(x), size=limit, replace=False))
        t_idx = torch.from_numpy(idx)
        x, y = x[t_idx], y[t_idx]
    return x, y


def decode(ids: list[int], itos: list[str]) -> str:
    return "".join("?" if i == 0 else itos[i] for i in ids)


class ResConv1dBlock(nn.Module):
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


class SeqTextAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        z_dim: int = 128,
        variational: bool = False,
        emb_dim: int = 64,
        d_model: int = 192,
        seq_len: int = WINDOW,
    ) -> None:
        super().__init__()
        if seq_len % 4 != 0:
            raise ValueError(f"seq_len debe ser múltiplo de 4, recibido {seq_len}")
        self.hparams = {
            "vocab_size": vocab_size,
            "z_dim": z_dim,
            "variational": variational,
            "emb_dim": emb_dim,
            "d_model": d_model,
            "seq_len": seq_len,
        }
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.variational = variational
        self.seq_len = seq_len
        self.down_len = seq_len // 4
        self.bottleneck_ch = 64

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.enc_in = nn.Conv1d(emb_dim, d_model, kernel_size=1)
        self.enc_blocks = nn.Sequential(
            ResConv1dBlock(d_model, kernel_size=5, dilation=1),
            ResConv1dBlock(d_model, kernel_size=5, dilation=2),
            ResConv1dBlock(d_model, kernel_size=5, dilation=4),
            ResConv1dBlock(d_model, kernel_size=3, dilation=1),
        )
        self.enc_down = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, self.bottleneck_ch, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        flat_dim = self.down_len * self.bottleneck_ch
        self.enc_proj = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.to_z = nn.Linear(256, z_dim)
        if variational:
            self.to_log_var = nn.Linear(256, z_dim)

        self.dec_from_z = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, flat_dim),
            nn.GELU(),
        )
        self.dec_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.bottleneck_ch, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dec_pos_emb = nn.Parameter(torch.randn(1, d_model, seq_len) * 0.02)
        self.dec_film = nn.Linear(z_dim, 2 * d_model)
        self.dec_blocks = nn.Sequential(
            ResConv1dBlock(d_model, kernel_size=5, dilation=1),
            ResConv1dBlock(d_model, kernel_size=5, dilation=2),
            ResConv1dBlock(d_model, kernel_size=5, dilation=4),
        )
        self.out_head = nn.Conv1d(d_model, vocab_size, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.embedding(x.long()).transpose(1, 2)
        h = self.enc_blocks(self.enc_in(h))
        h = self.enc_down(h).flatten(1)
        h = self.enc_proj(h)
        if self.variational:
            return self.to_z(h), self.to_log_var(h).clamp(-10.0, 10.0)
        return self.to_z(h), None

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_from_z(z).view(z.size(0), self.bottleneck_ch, self.down_len)
        h = self.dec_up(h) + self.dec_pos_emb
        gamma, beta = self.dec_film(z).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.dec_blocks(h)
        return self.out_head(h).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mu, log_var = self.encode(x)
        do_sample = self.training if sample_posterior is None else sample_posterior
        if self.variational and do_sample and log_var is not None:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return self.decode(z), mu, log_var


TextAutoencoder = SeqTextAutoencoder


def kl_divergence_per_dim(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())


def kl_divergence(
    mu: torch.Tensor, log_var: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    kl_dim = kl_divergence_per_dim(mu, log_var)
    if free_bits > 0.0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)
    return kl_dim.sum(dim=1)


def cyclical_kl_beta(
    epoch: int,
    total_epochs: int,
    target_beta: float,
    n_cycles: int = 4,
    ratio: float = 0.5,
) -> float:
    if target_beta <= 0.0:
        return 0.0
    cycle_len = max(1.0, total_epochs / float(n_cycles))
    pos = ((epoch - 1) % cycle_len) / cycle_len
    if pos < ratio:
        return float(target_beta * (pos / ratio))
    return float(target_beta)


def reconstruction_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), x.long(), reduction="none")
    return ce.sum(dim=1)


def load_model(path: Path, device: torch.device) -> tuple[SeqTextAutoencoder, dict[str, Any]]:
    state_dict, meta = load_checkpoint(path, device="cpu")
    model = SeqTextAutoencoder(**meta["hparams"])
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta


def load_w2_transformer(device: torch.device, path: Path = W2_BEST_CHECKPOINT) -> CharTransformer:
    state_dict, meta = load_checkpoint(path, device="cpu")
    model = CharTransformer(**meta["model_hparams"])
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.inference_mode()
def batched(fn, x: torch.Tensor, device: torch.device, batch_size: int = 1024) -> torch.Tensor:
    return torch.cat(
        [fn(x[i : i + batch_size].to(device)).float().cpu() for i in range(0, len(x), batch_size)],
        dim=0,
    )
