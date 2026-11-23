"""Inferencia rápida sobre checkpoints/model_best.pt con carga segura (weights_only=True)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint
from shared.config import SEED, WINDOW_SIZE
from shared.data import encode_text, load_split_df, read_vocab
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything
from shared.models import CharTransformer
from Workshop_2.src.train import BEST_CHECKPOINT

DEFAULT_TEXT = (
    "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough wi"
)


def load_model(path: Path = BEST_CHECKPOINT, device_name: str = "auto"):
    device, _ = resolve_device(device_name)
    state_dict, meta = load_checkpoint(path, device="cpu")
    model = CharTransformer(**meta["model_hparams"])
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta, device


@torch.no_grad()
def top5(model: CharTransformer, text: str, vocab: dict, device: torch.device):
    text = text[-WINDOW_SIZE:].rjust(WINDOW_SIZE)
    ids = torch.tensor([encode_text(text, vocab["stoi"])], device=device, dtype=torch.long)
    probs = model(ids).softmax(-1)[0].cpu()
    values, indices = probs.topk(5)
    return [(vocab["itos"][i], float(p.item())) for p, i in zip(values, indices, strict=True)]


def print_prediction(text: str, preds: list[tuple[str, float]], true: str | None = None) -> None:
    print(f"\ncontext: ...{text[-50:]!r}")
    if true is not None:
        print(f"true next char: {true!r}")
    for rank, (char, p) in enumerate(preds, 1):
        print(f"  {rank}. {char!r:8} {p:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    seed_everything(SEED)
    model, meta, device = load_model(args.checkpoint, args.device)
    vocab = read_vocab()
    print(
        f"Loaded {args.checkpoint.name}: run={meta.get('run_name')}, "
        f"epoch={meta.get('epoch')}, val_bpc={meta.get('val_bpc', float('nan')):.4f}"
    )

    test_df = load_split_df("test", columns=["x", "y"])
    rng = np.random.default_rng(SEED)
    for idx in rng.choice(len(test_df), size=5, replace=False):
        x, y = test_df["x"].iat[idx], test_df["y"].iat[idx]
        print_prediction(x, top5(model, x, vocab, device), true=y)

    print("\n=== hand-written text ===")
    print_prediction(args.text, top5(model, args.text, vocab, device))


if __name__ == "__main__":
    main()
