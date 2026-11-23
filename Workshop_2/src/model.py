"""Fachada de Workshop 2 sobre la arquitectura compartida CharTransformer (`shared/models.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import SEED, UNK_INDEX, W1_DIR, WINDOW_SIZE
from shared.data import encode_text, load_split_df, read_vocab
from shared.device import resolve_device, seed_everything
from shared.models import CharTransformer, count_parameters

WORKSHOP1 = W1_DIR
BEST_CHECKPOINT = ROOT / "checkpoints" / "model_best.pt"


def set_seeds(seed: int = SEED) -> None:
    seed_everything(seed, deterministic=True)


def get_device(preferred: str = "auto") -> torch.device:
    device, _ = resolve_device(preferred)
    return device


def load_split(name: str):
    return load_split_df(name)  # type: ignore[arg-type]


def main() -> None:
    set_seeds()
    vocab_size = int(read_vocab()["size"])
    model = CharTransformer(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=4)
    print(model)
    print(f"\nParámetros entrenables únicos: {count_parameters(model):,}")

    batch = torch.randint(0, vocab_size, (4, WINDOW_SIZE))
    model.eval()
    with torch.no_grad():
        logits_last = model(batch, return_all_tokens=False)
        logits_all = model(batch, return_all_tokens=True)
    print(f"Input shape:            {tuple(batch.shape)}")
    print(f"Next-char logits shape: {tuple(logits_last.shape)}")
    print(f"All-token logits shape: {tuple(logits_all.shape)}")


if __name__ == "__main__":
    main()
