"""Entrenamiento de WGAN-GP secuencial 1D con Gumbel-Softmax ST (Workshop 5 — Fase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    Critic,
    Generator,
    gumbel_temperature,
    load_training_data,
)
from Workshop_5.src.train_gan import quick_val_metrics

CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "workshop5" / "wgan_gp"


def gradient_penalty(
    critic: Critic, real: torch.Tensor, fake: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        real_f = real.float()
        fake_f = fake.detach().float()
        bs = real_f.size(0)
        alpha = torch.rand(bs, 1, 1, device=device, dtype=torch.float32)
        interpolated = (alpha * real_f + (1.0 - alpha) * fake_f).requires_grad_(True)
        scores = critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.reshape(bs, -1)
        grad_norm = torch.sqrt(gradients.pow(2).sum(dim=1) + 1e-12)
        return ((grad_norm - 1.0) ** 2).mean()


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
    critic = Critic(vocab_size=vocab_size, hidden_dim=args.critic_hidden_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.learning_rate, betas=(0.0, 0.9))
    writer = SummaryWriter(RUNS_DIR) if not args.fast_verify else None

    best_bigram_tv = float("inf")
    last_uniq = 0.0
    last_bi = 1.0

    for epoch in range(1, args.epochs + 1):
        tau = gumbel_temperature(epoch, args.epochs, args.tau_start, args.tau_end)
        generator.train()
        critic.train()
        total_g = 0.0
        total_c = 0.0
        batches_seen = 0
        g_steps = 0

        for batch_idx, real_ids in enumerate(loader, start=1):
            if args.max_batches and batch_idx > args.max_batches:
                break
            real_ids = real_ids.to(device, non_blocking=(device.type == "cuda"))
            real = F.one_hot(real_ids, num_classes=vocab_size).float()
            bs = real.size(0)

            opt_c.zero_grad(set_to_none=True)
            noise = torch.randn(bs, args.noise_dim, device=device)
            with torch.no_grad():
                fake = generator(noise, tau=tau, hard=True)
            gp = gradient_penalty(critic, real, fake, device)
            critic_loss = critic(fake).mean() - critic(real).mean() + args.lambda_gp * gp
            critic_loss.backward()
            _, fin_c = clip_grad_norm_safe(critic.parameters(), max_norm=10.0)
            if fin_c:
                opt_c.step()

            total_c += float(critic_loss.item())
            batches_seen += 1

            if batch_idx % args.critic_steps == 0:
                opt_g.zero_grad(set_to_none=True)
                noise = torch.randn(bs, args.noise_dim, device=device)
                generated = generator(noise, tau=tau, hard=True)
                g_loss = -critic(generated).mean()
                g_loss.backward()
                _, fin_g = clip_grad_norm_safe(generator.parameters(), max_norm=5.0)
                if fin_g:
                    opt_g.step()
                total_g += float(g_loss.item())
                g_steps += 1

        mean_c = total_c / max(1, batches_seen)
        mean_g = total_g / g_steps if g_steps > 0 else float("nan")
        uniq_ratio, bigram_tv = quick_val_metrics(generator, val_blocks, device, tau=tau)
        last_uniq, last_bi = uniq_ratio, bigram_tv

        if writer is not None:
            writer.add_scalar("loss/critic", mean_c, epoch)
            writer.add_scalar("loss/generator", mean_g, epoch)
            writer.add_scalar("val/unique_sequence_ratio", uniq_ratio, epoch)
            writer.add_scalar("val/bigram_tv", bigram_tv, epoch)
            writer.add_scalar("hparams/tau", tau, epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | tau {tau:.2f} | C {mean_c:.4f} | G {mean_g:.4f} "
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
        save_checkpoint(CHECKPOINT_DIR / "gan_wgan_last.pt", cpu_state_dict(generator), metadata=meta)
        if uniq_ratio >= 0.90 and bigram_tv < best_bigram_tv:
            best_bigram_tv = bigram_tv
            save_checkpoint(CHECKPOINT_DIR / "gan_wgan_best.pt", cpu_state_dict(generator), metadata=meta)

    if writer is not None:
        writer.close()

    if args.fast_verify:
        pen = 0.0 if last_uniq >= 0.95 else 1.0
        print(f"VAL_BIGRAM_TV={last_bi + pen:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena WGAN-GP 1D Gumbel-ST (W5 Fase 5).")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()

    if args.fast_verify and args.max_batches == 0:
        args.max_batches = 250
    train(args)


if __name__ == "__main__":
    main()
