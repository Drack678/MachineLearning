"""Evaluación rigurosa de GANs contra VAL/TEST + perplejidad bajo Transformer W2 (Fase 5)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_5.src.logging_utils import (  # noqa: E402
    configure_rocm_env,
    format_duration,
    log,
    silence_native_rocm_logs,
)

configure_rocm_env()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from shared.checkpoints import load_checkpoint  # noqa: E402
from shared.config import SEED, W2_BEST_CHECKPOINT  # noqa: E402
from shared.data import build_contiguous_blocks, read_vocab  # noqa: E402
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything  # noqa: E402
from shared.models import CharTransformer  # noqa: E402
from Workshop_5.src.gan import SEQUENCE_LENGTH, Generator  # noqa: E402

CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports" / "metrics"
SAMPLE_DIR = ROOT / "reports" / "generated_samples"


def distribution(counter: Counter) -> dict[Any, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def total_variation_distance(counter_a: Counter, counter_b: Counter) -> float:
    dist_a = distribution(counter_a)
    dist_b = distribution(counter_b)
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)


@torch.inference_mode()
def sample_sequences(
    generator: Generator,
    count: int,
    device: torch.device,
    *,
    batch_size: int = 128,
    tau: float = 0.7,
) -> torch.Tensor:
    generator.eval()
    chunks: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        bs = min(batch_size, remaining)
        noise = torch.randn(bs, generator.noise_dim, device=device)
        samples_oh = generator(noise, tau=tau, hard=True)
        ids = samples_oh.argmax(dim=-1).cpu().to(torch.uint8)
        chunks.append(ids)
        remaining -= bs
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def compute_lm_perplexity(
    sequences: torch.Tensor,
    device: torch.device,
    w2_ckpt: Path = W2_BEST_CHECKPOINT,
    batch_size: int = 256,
) -> float:
    if not w2_ckpt.exists():
        return float("nan")
    try:
        sd, meta = load_checkpoint(w2_ckpt, device="cpu")
        lm = CharTransformer(**meta["model_hparams"]).to(device).eval()
        lm.load_state_dict(sd)
    except Exception:
        return float("nan")

    total_nll = 0.0
    total_tok = 0
    for start in range(0, len(sequences), batch_size):
        blk = sequences[start : start + batch_size].to(device).long()
        xb, yb = blk[:, :-1], blk[:, 1:]
        logits = lm(xb, return_all_tokens=True)
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tok += yb.numel()
    mean_nll = total_nll / max(1, total_tok)
    return float(math.exp(min(mean_nll, 20.0)))


def load_generator_safe(checkpoint_path: Path, device: torch.device) -> tuple[Generator, dict[str, Any]]:
    state_dict, meta = load_checkpoint(checkpoint_path, device="cpu")
    gen = Generator(**meta["generator_hparams"])
    gen.load_state_dict(state_dict)
    return gen.to(device).eval(), meta


def ids_to_text(sequence_ids: list[int], itos: list[str]) -> str:
    return "".join("\ufffd" if itos[idx] == "<UNK>" else itos[idx] for idx in sequence_ids)


def get_metrics(
    generated_tokens: list[list[str]],
    real_tokens: list[list[str]],
    train_substrings: set[str] | None = None,
    unk_token: str = "<UNK>",
) -> dict[str, float]:
    unique_sequences = len({tuple(seq) for seq in generated_tokens})
    total_chars = sum(len(seq) for seq in generated_tokens)

    gen_chars = Counter(c for seq in generated_tokens for c in seq)
    real_chars = Counter(c for seq in real_tokens for c in seq)
    gen_bi = Counter((seq[i], seq[i + 1]) for seq in generated_tokens for i in range(len(seq) - 1))
    real_bi = Counter((seq[i], seq[i + 1]) for seq in real_tokens for i in range(len(seq) - 1))
    gen_tri = Counter(
        (seq[i], seq[i + 1], seq[i + 2]) for seq in generated_tokens for i in range(len(seq) - 2)
    )
    real_tri = Counter(
        (seq[i], seq[i + 1], seq[i + 2]) for seq in real_tokens for i in range(len(seq) - 2)
    )

    mem_rate = float("nan")
    if train_substrings is not None:
        hits = sum(1 for seq in generated_tokens if "".join(seq[:20]) in train_substrings)
        mem_rate = hits / max(1, len(generated_tokens))

    return {
        "unique_sequence_ratio": unique_sequences / max(1, len(generated_tokens)),
        "mean_unique_characters": sum(len(set(seq)) for seq in generated_tokens) / max(1, len(generated_tokens)),
        "character_distribution_tv": total_variation_distance(gen_chars, real_chars),
        "bigram_distribution_tv": total_variation_distance(gen_bi, real_bi),
        "trigram_distribution_tv": total_variation_distance(gen_tri, real_tri),
        "unknown_character_ratio": gen_chars.get(unk_token, 0) / max(1, total_chars),
        "memorization_20char_rate": mem_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa GANs contra val/test (Workshop 5 Fase 5).")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument(
        "--verbose-rocm",
        action="store_true",
        help="Muestra logs nativos de ROCm/MIOpen sin filtrar.",
    )
    args = parser.parse_args()

    with silence_native_rocm_logs(enabled=not args.verbose_rocm):
        t0 = time.time()
        seed_everything(args.seed, deterministic=True)
        device, desc = resolve_device(args.device)
        log(f"Device: {device} | {desc}")

        vocab = read_vocab()
        itos = vocab["itos"]
        real_blocks = build_contiguous_blocks(args.split, vocab=vocab, block_size=SEQUENCE_LENGTH, stride=80)
        n_real = min(args.num_samples, len(real_blocks))
        perm = torch.randperm(len(real_blocks), generator=torch.Generator().manual_seed(args.seed))[:n_real]
        real_seqs = real_blocks[perm].tolist()
        real_tokens = [[itos[i] for i in seq] for seq in real_seqs]

        train_blocks = build_contiguous_blocks("train", vocab=vocab, block_size=SEQUENCE_LENGTH, stride=80)
        train_substrings = {"".join(itos[i] for i in row[:20].tolist()) for row in train_blocks[:20_000]}

        models = {
            "vanilla": CHECKPOINT_DIR / "gan_vanilla_best.pt",
            "wgan_gp": CHECKPOINT_DIR / "gan_wgan_best.pt",
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

        rows = []
        for name, ckpt_path in models.items():
            if not ckpt_path.exists():
                ckpt_path = CHECKPOINT_DIR / f"gan_{'vanilla' if name == 'vanilla' else 'wgan'}_last.pt"
            if not ckpt_path.exists():
                log(f"[SKIP] No existe checkpoint para {name}: {ckpt_path}")
                continue

            t_m = time.time()
            log(f"Evaluando {name} ({ckpt_path.name}) con {args.num_samples:,} muestras...")
            gen, meta = load_generator_safe(ckpt_path, device)
            gen_ids = sample_sequences(gen, args.num_samples, device, tau=args.tau)
            gen_tokens = [[itos[i] for i in seq] for seq in gen_ids.tolist()]

            m = get_metrics(gen_tokens, real_tokens, train_substrings=train_substrings)
            m["lm_perplexity"] = compute_lm_perplexity(gen_ids, device)
            rows.append({"model": name, "split": args.split, "epoch": meta.get("epoch", -1), **m})

            sample_path = SAMPLE_DIR / f"{name}_samples.txt"
            with sample_path.open("w", encoding="utf-8") as f:
                for idx, seq in enumerate(gen_ids[:32].tolist(), start=1):
                    f.write(f"sample {idx}:\n{ids_to_text(seq, itos)}\n\n")

            log(
                f"[{name}] listo en {format_duration(time.time() - t_m)} | "
                f"uniq={m['unique_sequence_ratio']:.3f} | "
                f"bigram_tv={m['bigram_distribution_tv']:.4f} | "
                f"lm_ppl={m['lm_perplexity']:.2f}"
            )

            if name == "wgan_gp" and args.split == "val":
                print(f"VAL_BIGRAM_TV={m['bigram_distribution_tv']:.6f}", flush=True)

        if rows:
            results = pd.DataFrame(rows)
            results.to_csv(REPORT_DIR / "gan_quality.csv", index=False)
            log(f"Evaluación completada en {format_duration(time.time() - t0)}")
            print("\nCalidad de GANs (Workshop_5/reports/metrics/gan_quality.csv):", flush=True)
            print(results.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
