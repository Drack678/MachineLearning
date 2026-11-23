"""Generación de muestras desde un checkpoint seguro (weights_only=True) de Workshop 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import SEED
from shared.data import read_vocab
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything
from Workshop_5.src.evaluate_gan import ids_to_text, load_generator_safe, sample_sequences

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "gan_wgan_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera muestras de texto desde GAN 1D.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("num-samples debe ser positivo")
    if args.temperature <= 0:
        raise ValueError("temperature debe ser positiva")

    seed_everything(args.seed)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    vocab = read_vocab()
    gen, meta = load_generator_safe(args.checkpoint, device)
    seqs = sample_sequences(gen, args.num_samples, device, tau=args.temperature)
    print(f"Checkpoint: {args.checkpoint.name} (epoch={meta.get('epoch')}, tau={args.temperature})\n")
    for idx, seq in enumerate(seqs.tolist(), start=1):
        print(f"Sample {idx}:\n{ids_to_text(seq, vocab['itos'])}\n")


if __name__ == "__main__":
    main()
