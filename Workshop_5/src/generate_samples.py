"""Generación de muestras desde un checkpoint seguro (weights_only=True) de Workshop 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_5.src.logging_utils import (  # noqa: E402
    configure_rocm_env,
    log,
    silence_native_rocm_logs,
)

configure_rocm_env()

from shared.config import SEED  # noqa: E402
from shared.data import read_vocab  # noqa: E402
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything  # noqa: E402
from Workshop_5.src.evaluate_gan import ids_to_text, load_generator_safe, sample_sequences  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "gan_wgan_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera muestras de texto desde GAN 1D.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument(
        "--verbose-rocm",
        action="store_true",
        help="Muestra logs nativos de ROCm/MIOpen sin filtrar.",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("num-samples debe ser positivo")
    if args.temperature <= 0:
        raise ValueError("temperature debe ser positiva")

    with silence_native_rocm_logs(enabled=not args.verbose_rocm):
        seed_everything(args.seed)
        device, desc = resolve_device(args.device)
        log(f"Device: {device} | {desc}")

        vocab = read_vocab()
        gen, meta = load_generator_safe(args.checkpoint, device)
        seqs = sample_sequences(gen, args.num_samples, device, tau=args.temperature)
        log(
            f"Checkpoint: {args.checkpoint.name} "
            f"(epoch={meta.get('epoch')}, tau={args.temperature})"
        )
        print("", flush=True)
        for idx, seq in enumerate(seqs.tolist(), start=1):
            print(f"Sample {idx}:\n{ids_to_text(seq, vocab['itos'])}\n", flush=True)


if __name__ == "__main__":
    main()
