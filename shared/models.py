"""Arquitectura unificada CharTransformer compartida por Workshops 2, 3, 4 y 5."""

from __future__ import annotations

import math

import torch
from torch import nn

from shared.config import WINDOW_SIZE


class CharTransformer(nn.Module):
    """Transformer decoder-only a nivel de carácter con soporte multi-token y weight tying."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = WINDOW_SIZE,
        activation: str = "gelu",
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.hparams: dict[str, int | float | str | bool] = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "max_len": max_len,
            "activation": activation,
            "tie_weights": tie_weights,
        }
        self.max_len = max_len
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

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
        self.fc = nn.Linear(d_model, vocab_size)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(self._init_weights)
        if tie_weights:
            self.fc.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Longitud de secuencia {seq_len} supera max_len={self.max_len}")
        positions = torch.arange(seq_len, device=x.device)
        h = self.emb_dropout(self.token_embedding(x) + self.position_embedding(positions))
        mask = self.causal_mask[:seq_len, :seq_len]
        return self.blocks(h, mask=mask, is_causal=True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_sequence(x)[:, -1]

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        h = self._encode_sequence(x)
        if return_all_tokens:
            return self.fc(h)
        return self.fc(h[:, -1])


def count_parameters(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
