"""Data augmentation coherente y re-entrenamiento del Transformer (Workshop 5 — Fase 5)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint, save_checkpoint
from shared.config import SEED, W2_BEST_CHECKPOINT, WINDOW_SIZE, to_relative_path
from shared.data import build_contiguous_blocks, load_encoded_split, read_vocab
from shared.device import (
    DEVICE_CHOICES,
    PROFILE_CHOICES,
    amp_context,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    get_hardware_profile,
    resolve_device,
    seed_everything,
)
from shared.metrics import (
    apply_logit_adjustment,
    compute_multiclass_metrics,
    fit_logit_adjustment_tau,
)
from shared.models import CharTransformer, build_warmup_cosine_scheduler
from Workshop_2.src.train import predict_logits
from Workshop_5.src.evaluate_gan import load_generator_safe, sample_sequences
from Workshop_5.src.gan import SEQUENCE_LENGTH

OUTPUT_CHECKPOINT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports" / "metrics"
RUNS_DIR = ROOT / "runs" / "workshop5" / "augmentation"


@torch.inference_mode()
def generate_filtered_gan_blocks(
    gan_ckpt: Path,
    n_target: int,
    device: torch.device,
    w2_ckpt: Path = W2_BEST_CHECKPOINT,
    *,
    oversample_factor: int = 2,
    tau: float = 0.7,
) -> torch.Tensor:
    gen, _ = load_generator_safe(gan_ckpt, device)
    raw = sample_sequences(gen, n_target * oversample_factor, device, tau=tau)

    if not w2_ckpt.exists():
        return raw[:n_target]

    sd, meta = load_checkpoint(w2_ckpt, device="cpu")
    lm = CharTransformer(**meta["model_hparams"]).to(device).eval()
    lm.load_state_dict(sd)

    nlls: list[torch.Tensor] = []
    for start in range(0, len(raw), 256):
        blk = raw[start : start + 256].to(device).long()
        logits = lm(blk[:, :-1], return_all_tokens=True)
        per_seq = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), blk[:, 1:], reduction="none"
        ).mean(dim=1)
        nlls.append(per_seq.cpu())
    scores = torch.cat(nlls, dim=0)
    keep_idx = scores.argsort(descending=False)[:n_target]
    return raw[keep_idx]


def build_char_noise_augmented_blocks(
    real_blocks: torch.Tensor,
    train_char_probs: np.ndarray,
    n_aug: int,
    *,
    noise_prob: float = 0.015,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n_real = len(real_blocks)
    idx = rng.choice(n_real, size=min(n_aug, n_real), replace=(n_aug > n_real))
    chosen = real_blocks[torch.from_numpy(idx)].clone()

    x_aug = chosen[:, :-1].clone()
    y_aug = chosen[:, 1:].clone()

    mask = torch.from_numpy(rng.random(x_aug.shape) < noise_prob)
    sampled = torch.from_numpy(
        rng.choice(len(train_char_probs), size=x_aug.shape, p=train_char_probs).astype(np.uint8)
    )
    x_aug[mask] = sampled[mask]
    return x_aug, y_aug


def train_condition(
    condition: str,
    x_blocks: torch.Tensor,
    y_blocks: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model_hparams: dict[str, Any],
    train_class_counts: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(args.seed, deterministic=True)
    profile = get_hardware_profile(device, args.profile)
    batch_size = args.batch_size or profile.batch_size

    model = CharTransformer(**model_hparams).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        foreach=profile.use_foreach,
    )
    loader = DataLoader(
        TensorDataset(x_blocks, y_blocks),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **dataloader_kwargs(device, num_workers=args.num_workers),
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.05)
    scaler = torch.amp.GradScaler("cuda") if profile.use_grad_scaler else None
    writer = SummaryWriter(RUNS_DIR / condition)

    best_val_bpc = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches_ok = 0
        skipped = 0

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            xb = xb.to(device, non_blocking=(device.type == "cuda")).long()
            yb = yb.to(device, non_blocking=(device.type == "cuda")).long()

            with amp_context(device, enabled=profile.amp_enabled, dtype=profile.amp_dtype):
                logits = model(xb, return_all_tokens=True)
                loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            if not torch.isfinite(loss):
                skipped += 1
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if not is_fin:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    skipped += 1
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if not is_fin:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())
            batches_ok += 1

        if batches_ok == 0:
            raise RuntimeError(f"[{condition}] Todos los batches fueron no finitos en época {epoch}.")

        mean_loss = total_loss / batches_ok
        val_logits = predict_logits(
            model, x_val, device, amp_enabled=profile.amp_enabled, amp_dtype=profile.amp_dtype
        )
        val_m = compute_multiclass_metrics(y_val.numpy(), val_logits, with_ranking_metrics=False)

        row = {
            "condition": condition,
            "epoch": epoch,
            "train_bpc": mean_loss / math.log(2.0),
            "val_bpc": val_m["bpc"],
            "val_accuracy": val_m["accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "skipped_batches": skipped,
        }
        history.append(row)
        writer.add_scalar("train/bpc", row["train_bpc"], epoch)
        writer.add_scalar("val/bpc", row["val_bpc"], epoch)
        writer.add_scalar("val/accuracy", row["val_accuracy"], epoch)
        writer.add_scalar("val/f1_macro", row["val_f1_macro"], epoch)

        print(
            f"[{condition}] ep {epoch:02d}/{args.epochs} | train/val BPC "
            f"{row['train_bpc']:.4f}/{row['val_bpc']:.4f} | val acc {row['val_accuracy']:.4f} "
            f"| val F1 {row['val_f1_macro']:.4f} (skipped={skipped})"
        )

        if val_m["bpc"] < best_val_bpc - 1e-4:
            best_val_bpc = val_m["bpc"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[{condition}] early stop en época {epoch}")
                break

    writer.close()
    if best_state is None:
        raise RuntimeError(f"[{condition}] No se generó ningún checkpoint válido (best_state=None).")

    model.load_state_dict(best_state)
    val_logits = predict_logits(model, x_val, device, amp_enabled=profile.amp_enabled, amp_dtype=profile.amp_dtype)
    tau_opt, _ = fit_logit_adjustment_tau(val_logits, y_val.numpy(), train_class_counts)
    test_logits = predict_logits(model, x_test, device, amp_enabled=profile.amp_enabled, amp_dtype=profile.amp_dtype)

    val_m = compute_multiclass_metrics(y_val.numpy(), val_logits, with_ranking_metrics=False)
    test_cal = compute_multiclass_metrics(y_test.numpy(), test_logits, with_ranking_metrics=False)
    test_adj = compute_multiclass_metrics(
        y_test.numpy(),
        apply_logit_adjustment(test_logits, train_class_counts, tau_opt),
        with_ranking_metrics=False,
    )

    ckpt_path = OUTPUT_CHECKPOINT_DIR / f"transformer_{condition}.pt"
    save_checkpoint(
        ckpt_path,
        cpu_state_dict(model),
        metadata={
            "model_hparams": model_hparams,
            "run_name": f"workshop5_{condition}",
            "augmentation": condition,
            "epoch": best_epoch,
            "logit_adj_tau": tau_opt,
            "val_bpc": val_m["bpc"],
            "test_accuracy": test_cal["accuracy"],
            "test_f1_macro_cal": test_cal["f1_macro"],
            "test_f1_macro_adj": test_adj["f1_macro"],
        },
    )

    result = {
        "condition": condition,
        "train_blocks": len(x_blocks),
        "best_epoch": best_epoch,
        "val_bpc": val_m["bpc"],
        "val_accuracy": val_m["accuracy"],
        "test_bpc": test_cal["bpc"],
        "test_accuracy": test_cal["accuracy"],
        "test_f1_macro_cal": test_cal["f1_macro"],
        "test_f1_macro_adj": test_adj["f1_macro"],
        "logit_adj_tau": tau_opt,
        "checkpoint": to_relative_path(ckpt_path),
    }
    return result, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentation y re-entrenamiento W5 (Fase 5).")
    parser.add_argument("--synthetic-ratio", type=float, default=0.10)
    parser.add_argument("--noise-prob", type=float, default=0.015)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=100_000)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES)
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=True)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    vocab = read_vocab()
    vocab_size = int(vocab["size"])
    if W2_BEST_CHECKPOINT.exists():
        _, w2_meta = load_checkpoint(W2_BEST_CHECKPOINT, device="cpu")
        model_hparams = w2_meta["model_hparams"]
    else:
        model_hparams = {
            "vocab_size": vocab_size, "d_model": 256, "n_heads": 4,
            "n_layers": 4, "dropout": 0.1, "max_len": WINDOW_SIZE,
            "activation": "gelu", "tie_weights": True,
        }

    real_blocks = build_contiguous_blocks("train", vocab=vocab, block_size=SEQUENCE_LENGTH, stride=args.stride)
    x_real_b, y_real_b = real_blocks[:, :-1], real_blocks[:, 1:]

    x_val, y_val = load_encoded_split("val", vocab=vocab)
    if args.max_val_samples and args.max_val_samples < len(x_val):
        idx = np.sort(np.random.default_rng(args.seed).choice(len(x_val), args.max_val_samples, replace=False))
        x_val, y_val = x_val[torch.from_numpy(idx)], y_val[torch.from_numpy(idx)]

    x_test, y_test = load_encoded_split("test", vocab=vocab)
    if args.max_test_samples and args.max_test_samples < len(x_test):
        idx = np.sort(np.random.default_rng(args.seed).choice(len(x_test), args.max_test_samples, replace=False))
        x_test, y_test = x_test[torch.from_numpy(idx)], y_test[torch.from_numpy(idx)]

    _, y_train_full = load_encoded_split("train", vocab=vocab)
    train_counts = torch.bincount(y_train_full.long(), minlength=vocab_size).numpy()
    char_probs = train_counts.astype(np.float64)
    char_probs = char_probs / char_probs.sum()

    n_aug = max(1, int(round(len(real_blocks) * args.synthetic_ratio)))
    conditions: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("real_only", x_real_b, y_real_b),
    ]

    wgan_ckpt = OUTPUT_CHECKPOINT_DIR / "gan_wgan_best.pt"
    if not wgan_ckpt.exists():
        wgan_ckpt = OUTPUT_CHECKPOINT_DIR / "gan_wgan_last.pt"
    if wgan_ckpt.exists():
        syn_blocks = generate_filtered_gan_blocks(wgan_ckpt, n_aug, device=device)
        conditions.append(
            (
                "wgan_gp_filtered",
                torch.cat([x_real_b, syn_blocks[:, :-1]], dim=0),
                torch.cat([y_real_b, syn_blocks[:, 1:]], dim=0),
            )
        )

    x_noise, y_noise = build_char_noise_augmented_blocks(
        real_blocks, char_probs, n_aug, noise_prob=args.noise_prob, seed=args.seed
    )
    conditions.append(
        (
            "char_noise_aug",
            torch.cat([x_real_b, x_noise], dim=0),
            torch.cat([y_real_b, y_noise], dim=0),
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for cond_name, xb, yb in conditions:
        res, hist = train_condition(
            cond_name, xb, yb, x_val, y_val, x_test, y_test,
            model_hparams, train_counts, args, device,
        )
        results.append(res)
        histories.extend(hist)
        pd.DataFrame(results).to_csv(REPORT_DIR / "augmentation_results.csv", index=False)

    pd.DataFrame(histories).to_csv(REPORT_DIR / "augmentation_history.csv", index=False)
    print("\nResumen final de Data Augmentation (Workshop_5/reports/metrics/augmentation_results.csv):")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
