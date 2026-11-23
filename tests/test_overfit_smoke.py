"""Smoke tests de overfit de 1 batch."""

from __future__ import annotations

import torch
from torch import nn

from shared.models import CharTransformer


def test_char_transformer_overfits_single_batch(tiny_batch) -> None:
    x, y = tiny_batch
    model = CharTransformer(vocab_size=16, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = float("inf")
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, return_all_tokens=True)
        loss = criterion(logits.reshape(-1, 16), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    assert final_loss < 0.05, f"El Transformer no logró overfit en 1 batch: loss={final_loss:.4f}"
