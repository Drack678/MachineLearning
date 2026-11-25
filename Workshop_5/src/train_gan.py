"""Entrenamiento de Vanilla GAN secuencial 1D con Gumbel-Softmax ST y Spectral Norm (Fase 5)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from shared.checkpoints import save_checkpoint  # noqa: E402
from shared.config import SEED, to_relative_path  # noqa: E402
from shared.data import build_contiguous_blocks  # noqa: E402
from shared.device import (  # noqa: E402
    DEVICE_CHOICES,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    is_rocm_build,
    resolve_device,
    seed_everything,
)
from Workshop_5.src.gan import (  # noqa: E402
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
    log(f"Device: {device} | {desc}")

    t_data = time.time()
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
    n_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    log(
        f"Datos listos en {format_duration(time.time() - t_data)} | "
        f"train={len(dataset):,} bloques | val={len(val_blocks):,} bloques"
    )

    vocab_size = int(vocab["size"])
    generator = Generator(
        vocab_size=vocab_size, noise_dim=args.noise_dim, hidden_dim=args.hidden_dim
    ).to(device)
    discriminator = Discriminator(vocab_size=vocab_size, hidden_dim=args.disc_hidden_dim).to(device)
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())

    use_foreach = False if is_rocm_build() else None
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), foreach=use_foreach
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), foreach=use_foreach
    )
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(RUNS_DIR)

    log(
        f"--- Vanilla GAN | G={g_params / 1e6:.2f}M D={d_params / 1e6:.2f}M params | "
        f"epochs={args.epochs} | batch={args.batch_size} ({n_batches} steps/ep) | "
        f"foreach={use_foreach} ---"
    )

    best_bigram_tv = float("inf")
    epoch_times: list[float] = []
    t_all = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        last_heartbeat = ep_t0
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

            now = time.time()
            if epoch == 1 and batch_idx == 1:
                log(
                    f"[vanilla] warmup GPU listo ({format_duration(now - ep_t0)}) | "
                    f"entrenando {n_batches} batches/ep..."
                )
                last_heartbeat = now
            elif batch_idx < n_batches and (now - last_heartbeat) >= 15.0:
                pct = 100.0 * batch_idx / max(1, n_batches)
                log(
                    f"[vanilla] ep {epoch:02d}/{args.epochs:02d} | "
                    f"batch {batch_idx}/{n_batches} ({pct:4.1f}%) | "
                    f"G={total_g / batches_seen:.4f} | D={total_d / batches_seen:.4f} | "
                    f"elapsed={format_duration(now - ep_t0)}"
                )
                last_heartbeat = now

        mean_g = total_g / max(1, batches_seen)
        mean_d = total_d / max(1, batches_seen)
        uniq_ratio, bigram_tv = quick_val_metrics(generator, val_blocks, device, tau=tau)

        writer.add_scalar("loss/generator", mean_g, epoch)
        writer.add_scalar("loss/discriminator", mean_d, epoch)
        writer.add_scalar("val/unique_sequence_ratio", uniq_ratio, epoch)
        writer.add_scalar("val/bigram_tv", bigram_tv, epoch)
        writer.add_scalar("hparams/tau", tau, epoch)

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
        is_best = uniq_ratio >= 0.80 and bigram_tv < best_bigram_tv
        if is_best:
            best_bigram_tv = bigram_tv
            save_checkpoint(CHECKPOINT_DIR / "gan_vanilla_best.pt", cpu_state_dict(generator), metadata=meta)

        ep_s = time.time() - ep_t0
        epoch_times.append(ep_s)
        avg_ep_s = (
            sum(epoch_times[1:]) / len(epoch_times[1:]) if len(epoch_times) > 1 else epoch_times[0]
        )
        eta_s = avg_ep_s * (args.epochs - epoch)
        best_tag = " *best" if is_best else ""
        log(
            f"[vanilla] ep {epoch:02d}/{args.epochs:02d} | tau={tau:.2f} | "
            f"G={mean_g:.4f} | D={mean_d:.4f} | "
            f"val_uniq={uniq_ratio:.3f} | val_bigram_tv={bigram_tv:.4f} | "
            f"{format_duration(ep_s)} | ETA {format_duration(eta_s)}{best_tag}"
        )

    writer.close()
    log(
        f"Vanilla GAN finalizado en {format_duration(time.time() - t_all)} | "
        f"best_bigram_tv={best_bigram_tv:.4f} | "
        f"ckpt={to_relative_path(CHECKPOINT_DIR / 'gan_vanilla_best.pt')}"
    )


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
    parser.add_argument(
        "--verbose-rocm",
        action="store_true",
        help="Muestra logs nativos de ROCm/MIOpen sin filtrar.",
    )
    args = parser.parse_args()
    with silence_native_rocm_logs(enabled=not args.verbose_rocm):
        train(args)


if __name__ == "__main__":
    main()
