"""Fixtures compartidas para la suite de pruebas del repositorio."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from shared.device import seed_everything


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    seed_everything(42, deterministic=True)


@pytest.fixture()
def synthetic_vocab() -> dict:
    chars = ["<UNK>", " ", "a", "b", "c", "e", "h", "t", "A", "."]
    return {
        "size": len(chars),
        "itos": chars,
        "stoi": {c: i for i, c in enumerate(chars)},
        "unk_index": 0,
    }


@pytest.fixture()
def synthetic_blocks_df() -> pd.DataFrame:
    rows = []
    texts = {
        "CHAR_A": "the cat sat on the mat. the bat sat on the hat. " * 4,
        "CHAR_B": "a cab had a bad hat. a bad cab had a bat. " * 4,
    }
    for char_id, text in texts.items():
        for i in range(len(text) - 80):
            rows.append({"character_id": char_id, "x": text[i : i + 80], "y": text[i + 80]})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    x = torch.randint(1, 16, (8, 80), generator=g)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y
