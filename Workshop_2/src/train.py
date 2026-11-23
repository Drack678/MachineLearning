"""Entrenamiento autorregresivo multi-token del Transformer de Workshop 2 (Fase 2)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from shared.checkpoints import save_checkpoint
from shared.config import SEED, WINDOW_SIZE, to_relative_path
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
from shared.models import CharTransformer, build_warmup_cosine_scheduler, count_parameters

CHECKPOINTS_DIR = ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINTS_DIR / "model_best.pt"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEARCH_LRS = (1.5e-3, 1e-3, 5e-4)
SEARCH_REG = (
    {"dropout": 0.1, "weight_decay": 0.05},
    {"dropout": 0.2, "weight_decay": 0.10},
)


def subsample_indices(n: int, size: int | None, seed: int = SEED) -> np.ndarray | None:
    if size is None or size >= n:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device, non_blocking=(device.type == "cuda")).long()
        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xb, return_all_tokens=False)
        chunks.append(logits.float().cpu())
    return torch.cat(chunks, dim=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    grad_accum: int = 1,
    clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(loader)
    for step_idx, (blocks,) in enumerate(loader, start=1):
        blocks = blocks.to(device, non_blocking=(device.type == "cuda")).long()
        xb = blocks[:, :-1]
        yb = blocks[:, 1:]

        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xb, return_all_tokens=True)
            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss = raw_loss / float(grad_accum)

        if not torch.isfinite(raw_loss):
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step_idx % grad_accum == 0) or (step_idx == num_batches)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            _, is_finite = clip_grad_norm_safe(model.parameters(), max_norm=clip_norm)
            if not is_finite:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                skipped += 1
                continue
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        n_tok = yb.numel()
        total_loss += float(raw_loss.item()) * n_tok
        total_tokens += n_tok

    mean_nll = total_loss / max(total_tokens, 1)
    return {
        "train_nll": mean_nll,
        "train_bpc": mean_nll / math.log(2.0),
        "skipped_batches": float(skipped),
    }


def plot_learning_curves(history: list[dict[str, Any]], run_name: str) -> None:
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h["epoch"], h["train_bpc"], "o-", label="train BPC")
    axes[0].plot(h["epoch"], h["val_bpc"], "s-", label="val BPC")
    axes[0].set_title("Bits per Character (lower is better)")
    axes[1].plot(h["epoch"], h["val_accuracy"], "s-", label="val accuracy")
    axes[1].plot(h["epoch"], h["val_top5_accuracy"], "^-", label="val top-5")
    axes[1].set_title("Validation Accuracy")
    axes[2].plot(h["epoch"], h["val_f1_macro"], "s-", label="val F1-macro (tau=0)")
    axes[2].set_title("Validation F1-macro")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(run_name)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"learning_curves_{run_name}.png", dpi=110)
    plt.close(fig)


def train_model(
    args: argparse.Namespace,
    vocab: dict[str, Any],
    train_blocks: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    train_class_counts: np.ndarray,
    device: torch.device,
    *,
    save: bool = True,
) -> dict[str, Any]:
    seed_everything(args.seed, deterministic=True)
    profile = get_hardware_profile(device, args.profile)
    batch_size = args.batch_size or profile.batch_size
    grad_accum = args.grad_accum or profile.grad_accum

    vocab_size = int(vocab["size"])
    model = CharTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=WINDOW_SIZE,
        activation="gelu",
        tie_weights=True,
    ).to(device)

    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_blocks),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_gen,
        **dataloader_kwargs(device, num_workers=args.num_workers),
    )

    steps_per_epoch = math.ceil(len(loader) / float(grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        foreach=profile.use_foreach,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio, min_lr_ratio=0.01
    )
    scaler = torch.amp.GradScaler("cuda") if profile.use_grad_scaler else None
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(RUNS_DIR / args.run_name) if not args.fast_verify else None

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    t_start = time.time()

    print(
        f"[{args.run_name}] params={count_parameters(model):,} | blocks={len(train_blocks):,} "
        f"| val={len(val_x):,} | device={device} ({profile.name}) | bs={batch_size}x{grad_accum}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            criterion,
            device,
            amp_enabled=profile.amp_enabled,
            amp_dtype=profile.amp_dtype,
            scaler=scaler,
            grad_accum=grad_accum,
            clip_norm=args.clip_norm,
        )

        val_logits = predict_logits(
            model,
            val_x,
            device,
            batch_size=max(512, batch_size * 4),
            amp_enabled=profile.amp_enabled,
            amp_dtype=profile.amp_dtype,
        )
        val_m = compute_multiclass_metrics(
            val_y.numpy(),
            val_logits,
            with_ranking_metrics=False,
        )
        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_nll": tr["train_nll"],
            "train_bpc": tr["train_bpc"],
            "val_nll": val_m["cross_entropy_nats"],
            "val_bpc": val_m["bpc"],
            "val_accuracy": val_m["accuracy"],
            "val_top5_accuracy": val_m["top5_accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "val_ece": val_m["ece"],
            "skipped_batches": tr["skipped_batches"],
        }
        history.append(row)
        if writer is not None:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)

        improved = best is None or row["val_bpc"] < best["val_bpc"] - 1e-4
        print(
            f"epoch {epoch:02d}/{args.epochs} | train/val BPC {row['train_bpc']:.4f}/{row['val_bpc']:.4f} "
            f"| val acc {row['val_accuracy']:.4f} | top5 {row['val_top5_accuracy']:.4f} "
            f"| F1-macro {row['val_f1_macro']:.4f} | lr {lr_now:.2e} ({time.time() - t0:.1f}s)"
            f"{' *best*' if improved else ''}"
        )
        if improved:
            best = dict(row)
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping en época {epoch} (sin mejora en {args.patience} épocas).")
                break

    if writer is not None:
        writer.close()

    if best_state is None or best is None:
        raise RuntimeError("No se obtuvo ningún checkpoint válido durante el entrenamiento.")

    model.load_state_dict(best_state)
    best_val_logits = predict_logits(
        model,
        val_x,
        device,
        batch_size=max(512, batch_size * 4),
        amp_enabled=profile.amp_enabled,
        amp_dtype=profile.amp_dtype,
    )
    tau_opt, val_f1_adj = fit_logit_adjustment_tau(best_val_logits, val_y.numpy(), train_class_counts)
    val_metrics_raw = compute_multiclass_metrics(
        val_y.numpy(), best_val_logits, with_ranking_metrics=not args.fast_verify
    )
    val_metrics_adj = compute_multiclass_metrics(
        val_y.numpy(),
        apply_logit_adjustment(best_val_logits, train_class_counts, tau_opt),
        with_ranking_metrics=False,
    )

    result = {
        "history": history,
        "best": best,
        "logit_adj_tau": tau_opt,
        "val_metrics_calibrated": val_metrics_raw,
        "val_metrics_logit_adjusted": val_metrics_adj,
        "train_seconds": time.time() - t_start,
    }

    if args.fast_verify:
        print(f"VAL_BPC={best['val_bpc']:.6f}")
        print(f"VAL_F1_MACRO_ADJ={val_f1_adj:.6f}")

    if save and not args.fast_verify:
        plot_learning_curves(history, args.run_name)
        metadata = {
            "model_hparams": model.hparams,
            "train_hparams": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "epochs": args.epochs,
                "stride": args.stride,
                "warmup_ratio": args.warmup_ratio,
                "clip_norm": args.clip_norm,
                "patience": args.patience,
                "seed": args.seed,
                "profile": profile.name,
                "objective": "autoregressive_multi_token_ce_calibrated",
            },
            "epoch": best["epoch"],
            "val_bpc": best["val_bpc"],
            "logit_adj_tau": tau_opt,
            "train_class_counts": train_class_counts.tolist(),
            "val_metrics_calibrated": val_metrics_raw,
            "val_metrics_logit_adjusted": val_metrics_adj,
            "run_name": args.run_name,
            "train_seconds": result["train_seconds"],
            "history": history,
        }
        ckpt_path = CHECKPOINTS_DIR / (
            "model_best.pt" if args.save_as_best else f"{args.run_name}.pt"
        )
        save_checkpoint(ckpt_path, cpu_state_dict(model), metadata=metadata)
        print(
            f"Checkpoint seguro guardado en {to_relative_path(ckpt_path)} "
            f"(epoch={best['epoch']}, val_bpc={best['val_bpc']:.4f}, "
            f"tau={tau_opt:.2f}, val_f1_adj={val_f1_adj:.4f})"
        )

    return result


def run_search(
    args: argparse.Namespace,
    vocab: dict[str, Any],
    train_blocks: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    train_class_counts: np.ndarray,
    device: torch.device,
) -> None:
    rows = []
    for lr in SEARCH_LRS:
        for reg in SEARCH_REG:
            run_args = argparse.Namespace(**vars(args))
            run_args.lr = lr
            run_args.dropout = reg["dropout"]
            run_args.weight_decay = reg["weight_decay"]
            run_args.run_name = f"search_v2_lr{lr:g}_do{reg['dropout']:g}_wd{reg['weight_decay']:g}"
            res = train_model(
                run_args, vocab, train_blocks, val_x, val_y, train_class_counts, device, save=False
            )
            best = res["best"]
            rows.append(
                {
                    "run_name": run_args.run_name,
                    "lr": lr,
                    "dropout": reg["dropout"],
                    "weight_decay": reg["weight_decay"],
                    "best_epoch": best["epoch"],
                    "val_bpc": best["val_bpc"],
                    "val_accuracy": best["val_accuracy"],
                    "val_f1_macro": best["val_f1_macro"],
                    "val_f1_macro_adj": res["val_metrics_logit_adjusted"]["f1_macro"],
                    "logit_adj_tau": res["logit_adj_tau"],
                    "minutes": res["train_seconds"] / 60.0,
                }
            )
    table = pd.DataFrame(rows).sort_values("val_bpc", ascending=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "hparam_search.csv", index=False)
    print(table.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento multi-token del Transformer W2.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max-train-blocks", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES)
    parser.add_argument("--run-name", default="transformer_v2")
    parser.add_argument("--save-as-best", action="store_true", default=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()

    if args.fast_verify:
        args.max_train_blocks = args.max_train_blocks or 12_000
        args.max_val_samples = args.max_val_samples or 50_000

    seed_everything(args.seed, deterministic=True)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    vocab = read_vocab()
    train_blocks = build_contiguous_blocks(
        "train", vocab=vocab, block_size=WINDOW_SIZE + 1, stride=args.stride
    )
    if args.max_train_blocks is not None and args.max_train_blocks < len(train_blocks):
        idx = subsample_indices(len(train_blocks), args.max_train_blocks, seed=args.seed)
        train_blocks = train_blocks[torch.from_numpy(idx)]

    val_x, val_y = load_encoded_split("val", vocab=vocab)
    if args.max_val_samples is not None and args.max_val_samples < len(val_x):
        v_idx = subsample_indices(len(val_x), args.max_val_samples, seed=args.seed)
        val_x = val_x[torch.from_numpy(v_idx)]
        val_y = val_y[torch.from_numpy(v_idx)]

    _, train_y_full = load_encoded_split("train", vocab=vocab)
    train_class_counts = torch.bincount(train_y_full.long(), minlength=int(vocab["size"])).numpy()

    if args.search:
        run_search(args, vocab, train_blocks, val_x, val_y, train_class_counts, device)
    else:
        train_model(args, vocab, train_blocks, val_x, val_y, train_class_counts, device, save=True)


if __name__ == "__main__":
    main()
