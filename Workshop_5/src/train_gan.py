"""Entrenamiento de Vanilla GAN secuencial 1D con Gumbel-Softmax ST y Spectral Norm (Fase 5)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import save_checkpoint
from shared.config import SEED
from shared.data import build_contiguous_blocks
from shared.device import (
    DEVICE_CHOICES,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    resolve_device,
    seed_everything,
)
from Workshop_5.src.gan import (
    SEQUENCE_LENGTH,
    Discriminator,
    Generator,
    gumbel_temperature,
    load_training_data,
)

CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "workshop5" / "vanilla"


def quick_val_metrics(
    generator: Generator,
    val_blocks: torch.Tensor,
    device: torch.device,
    n_samples: int = 500,
    tau: float = 0.7,
) -> tuple[float, float]:
    generator.eval()
    gen_ids: list[list[int]] = []
    with torch.inference_mode():
        rem = n_samples
        while rem > 0:
            bs = min(128, rem)
            z = torch.randn(bs, generator.noise_dim, device=device)
            logits = generator(z, tau=tau, hard=True, return_logits=True)
            ids = logits.argmax(dim=-1).cpu().tolist()
            gen_ids.extend(ids)
            rem -= bs

    unique_ratio = len({tuple(s) for s in gen_ids}) / max(1, len(gen_ids))
    real_ids = val_blocks[:n_samples].tolist()
    g_bi = Counter((s[i], s[i + 1]) for s in gen_ids for i in range(len(s) - 1))
    r_bi = Counter((s[i], s[i + 1]) for s in real_ids for i in range(len(s) - 1))
    gt = sum(g_bi.values()) or 1
    rt = sum(r_bi.values()) or 1
    keys = set(g_bi) | set(r_bi)
    bigram_tv = 0.5 * sum(abs(g_bi.get(k, 0) / gt - r_bi.get(k, 0) / rt) for k in keys)
    return float(unique_ratio), float(bigram_tv)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed, deterministic=True)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    dataset, vocab = load_training_data(stride=args.stride)
    val_blocks = build_contiguous_blocks("val", vocab=vocab, block_size=SEQUENCE_LENGTH, stride=80)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **dataloader_kwargs(device, num_workers=args.num_workers),
    )

    vocab_size = int(vocab["size"])
    generator = Generator(
        vocab_size=vocab_size, noise_dim=args.noise_dim, hidden_dim=args.hidden_dim
    ).to(device)
    discriminator = Discriminator(vocab_size=vocab_size, hidden_dim=args.disc_hidden_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(RUNS_DIR)

    best_bigram_tv = float("inf")
    for epoch in range(1, args.epochs + 1):
        tau = gumbel_temperature(epoch, args.epochs, args.tau_start, args.tau_end)
        generator.train()
        discriminator.train()
        total_g = 0.0
        total_d = 0.0
        batches_seen = 0

        for batch_idx, ids in enumerate(loader, start=1):
            if args.max_batches and batch_idx > args.max_batches:
                break
            ids = ids.to(device, non_blocking=(device.type == "cuda"))
            real = F.one_hot(ids, num_classes=vocab_size).float()
            bs = real.size(0)

            opt_d.zero_grad(set_to_none=True)
            noise = torch.randn(bs, args.noise_dim, device=device)
            with torch.no_grad():
                fake = generator(noise, tau=tau, hard=True)
            real_logits = discriminator(real)
            fake_logits = discriminator(fake)
            d_loss = criterion(real_logits, torch.full_like(real_logits, 0.9)) + criterion(
                fake_logits, torch.zeros_like(fake_logits)
            )
            d_loss.backward()
            _, fin_d = clip_grad_norm_safe(discriminator.parameters(), max_norm=5.0)
            if fin_d:
                opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            noise = torch.randn(bs, args.noise_dim, device=device)
            generated = generator(noise, tau=tau, hard=True)
            g_loss = criterion(discriminator(generated), torch.ones(bs, device=device))
            g_loss.backward()
            _, fin_g = clip_grad_norm_safe(generator.parameters(), max_norm=5.0)
            if fin_g:
                opt_g.step()

            total_g += float(g_loss.item())
            total_d += float(d_loss.item())
            batches_seen += 1

        mean_g = total_g / max(1, batches_seen)
        mean_d = total_d / max(1, batches_seen)
        uniq_ratio, bigram_tv = quick_val_metrics(generator, val_blocks, device, tau=tau)

        writer.add_scalar("loss/generator", mean_g, epoch)
        writer.add_scalar("loss/discriminator", mean_d, epoch)
        writer.add_scalar("val/unique_sequence_ratio", uniq_ratio, epoch)
        writer.add_scalar("val/bigram_tv", bigram_tv, epoch)
        writer.add_scalar("hparams/tau", tau, epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | tau {tau:.2f} | G {mean_g:.4f} | D {mean_d:.4f} "
            f"| val_uniq {uniq_ratio:.3f} | val_bigram_tv {bigram_tv:.4f}"
        )

        meta = {
            "epoch": epoch,
            "tau": tau,
            "generator_hparams": generator.hparams,
            "val_unique_sequence_ratio": uniq_ratio,
            "val_bigram_tv": bigram_tv,
            "vocab_size": vocab_size,
            "noise_dim": args.noise_dim,
        }
        save_checkpoint(CHECKPOINT_DIR / "gan_vanilla_last.pt", cpu_state_dict(generator), metadata=meta)
        if uniq_ratio >= 0.80 and bigram_tv < best_bigram_tv:
            best_bigram_tv = bigram_tv
            save_checkpoint(CHECKPOINT_DIR / "gan_vanilla_best.pt", cpu_state_dict(generator), metadata=meta)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Vanilla GAN 1D Gumbel-ST (W5 Fase 5).")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--disc-hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
