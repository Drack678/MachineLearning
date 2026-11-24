"""Entrenamiento de AE y VAE secuenciales con cyclical KL annealing + free bits (Fase 4)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_4.src.logging_utils import (  # noqa: E402
    configure_rocm_env,
    format_duration,
    log,
    silence_native_rocm_logs,
)

configure_rocm_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from shared.checkpoints import save_checkpoint  # noqa: E402
from shared.config import SEED, to_relative_path  # noqa: E402
from shared.data import read_vocab  # noqa: E402
from shared.device import (  # noqa: E402
    DEVICE_CHOICES,
    PROFILE_CHOICES,
    amp_context,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    get_hardware_profile,
    is_rocm_build,
    resolve_device,
    seed_everything,
)
from Workshop_4.src.autoencoder import (  # noqa: E402
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    SeqTextAutoencoder,
    batched,
    cyclical_kl_beta,
    kl_divergence,
    load_split,
    reconstruction_loss,
)

DEFAULT_EPOCHS = 40
DEFAULT_LR = 1e-3
DEFAULT_FREE_BITS = 0.25
N_SILHOUETTE = 2000
MIN_KL_HEALTHY = 10.0


def top5_sample(y: torch.Tensor, n: int = N_SILHOUETTE) -> np.ndarray:
    top5 = torch.bincount(y.long()).argsort(descending=True)[:5]
    rows = torch.nonzero(torch.isin(y.long(), top5)).squeeze(1).numpy()
    return np.sort(np.random.default_rng(SEED).choice(rows, min(n, len(rows)), replace=False))


@torch.inference_mode()
def evaluate_ae(
    model: SeqTextAutoencoder,
    x: torch.Tensor,
    device: torch.device,
    *,
    beta: float = 1.0,
    mc_samples: int = 4,
    batch_size: int = 1024,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    model.eval()
    ce_mu = 0.0
    ce_mc = 0.0
    correct_mu = 0
    kl_raw_sum = 0.0
    n = len(x)
    seq_len = x.size(1)

    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size].to(device, non_blocking=(device.type == "cuda")).long()
        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            logits_mu, mu, log_var = model(xb, sample_posterior=False)
        ce_mu += float(reconstruction_loss(logits_mu.float(), xb).sum().item())
        correct_mu += int((logits_mu.argmax(-1) == xb).sum().item())

        if model.variational and log_var is not None:
            kl_raw_sum += float(kl_divergence(mu.float(), log_var.float(), free_bits=0.0).sum().item())
            mc_acc = 0.0
            for _ in range(mc_samples):
                z_s = model.reparameterize(mu.float(), log_var.float())
                with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                    logits_s = model.decode(z_s)
                mc_acc += float(reconstruction_loss(logits_s.float(), xb).sum().item())
            ce_mc += mc_acc / float(mc_samples)
        else:
            ce_mc += float(reconstruction_loss(logits_mu.float(), xb).sum().item())

    recon_ce_mu = ce_mu / float(n * seq_len)
    recon_ce_mc = ce_mc / float(n * seq_len)
    kl_mean = kl_raw_sum / float(n)
    elbo_mc = (ce_mc / float(n)) + beta * kl_mean
    return {
        "recon_ce": recon_ce_mu,
        "recon_ce_mc": recon_ce_mc,
        "char_acc": correct_mu / float(n * seq_len),
        "kl": kl_mean,
        "elbo_mc": elbo_mc,
    }


def train_run(
    name: str,
    z_dim: int,
    variational: bool,
    beta: float,
    data: dict[str, Any],
    vocab_size: int,
    device: torch.device,
    args: argparse.Namespace,
    *,
    run_idx: int = 1,
    total_runs: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seed_everything(args.seed, deterministic=True)
    profile = get_hardware_profile(device, args.profile)
    batch_size = args.batch_size or profile.batch_size

    x_train: torch.Tensor = data["x_train"]
    x_val: torch.Tensor = data["x_val"]
    seq_len = int(x_train.size(1))

    model = SeqTextAutoencoder(
        vocab_size=vocab_size, z_dim=z_dim, variational=variational
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=profile.use_foreach,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda") if profile.use_grad_scaler else None
    writer = SummaryWriter(RUNS_DIR / name) if not args.fast_verify else None

    loader = DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **dataloader_kwargs(device, num_workers=args.num_workers),
    )
    n_batches = len(loader)
    amp_label = str(profile.amp_dtype).replace("torch.", "") if profile.amp_enabled else "fp32"
    model_kind = f"VAE(z={z_dim}, beta={beta:g})" if variational else f"AE(z={z_dim})"
    log(
        f"--- [{run_idx}/{total_runs}] {name} | {model_kind} | {n_params / 1e6:.2f}M params "
        f"| epochs={args.epochs} | batch={batch_size} ({n_batches} steps/ep) | amp={amp_label} ---"
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    epoch_times: list[float] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        last_heartbeat = ep_t0
        model.train()
        beta_now = (
            cyclical_kl_beta(epoch, args.epochs, beta, n_cycles=args.kl_cycles)
            if variational
            else 0.0
        )
        ce_char_sum_t = torch.zeros((), device=device, dtype=torch.float32)
        n_seen = 0
        skipped_steps = 0

        for step, (xb,) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=(device.type == "cuda")).long()
            optimizer.zero_grad(set_to_none=True)

            with amp_context(device, enabled=profile.amp_enabled, dtype=profile.amp_dtype):
                logits, mu, log_var = model(xb, sample_posterior=True)
                recon = reconstruction_loss(logits.float(), xb)
                loss = recon.mean()
                if variational and log_var is not None:
                    kl_obj = kl_divergence(mu.float(), log_var.float(), free_bits=args.free_bits).mean()
                    loss = loss + beta_now * kl_obj

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if is_fin:
                    scaler.step(optimizer)
                else:
                    skipped_steps += 1
                scaler.update()
            else:
                loss.backward()
                _, is_fin = clip_grad_norm_safe(model.parameters(), max_norm=1.0)
                if is_fin:
                    optimizer.step()
                else:
                    skipped_steps += 1

            ce_char_sum_t = ce_char_sum_t + (recon.detach().float().sum() / float(seq_len))
            n_seen += xb.size(0)

            now = time.time()
            if epoch == 1 and step == 1:
                log(
                    f"[{name}] warmup GPU listo ({format_duration(now - ep_t0)}) | "
                    f"entrenando {n_batches} batches/ep..."
                )
                last_heartbeat = now
            elif step < n_batches and (now - last_heartbeat) >= 15.0:
                pct = 100.0 * step / max(1, n_batches)
                cur_ce = float(ce_char_sum_t.item()) / max(1, n_seen)
                log(
                    f"[{name}] ep {epoch:02d}/{args.epochs:02d} | "
                    f"batch {step}/{n_batches} ({pct:4.1f}%) | train_ce={cur_ce:.4f} | "
                    f"elapsed={format_duration(now - ep_t0)}"
                )
                last_heartbeat = now

        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        train_recon_ce = float(ce_char_sum_t.item()) / max(1, n_seen)
        val = evaluate_ae(
            model,
            x_val,
            device,
            beta=beta,
            mc_samples=2 if args.fast_verify else 4,
            amp_enabled=profile.amp_enabled,
            amp_dtype=profile.amp_dtype,
        )
        row = {
            "epoch": epoch,
            "train_recon_ce": train_recon_ce,
            "val_recon_ce": val["recon_ce"],
            "val_recon_ce_mc": val["recon_ce_mc"],
            "val_char_acc": val["char_acc"],
            "val_kl": val["kl"],
            "val_elbo_mc": val["elbo_mc"],
            "beta_now": beta_now,
        }
        history.append(row)
        if writer is not None:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)

        if variational:
            healthy = val["kl"] >= MIN_KL_HEALTHY
            score = val["elbo_mc"] if healthy else (val["elbo_mc"] + 1e4)
        else:
            score = val["recon_ce"]

        is_best = best is None or score < best["select_score"]
        if is_best:
            best = {"select_score": score, **row}
            best_state = cpu_state_dict(model)

        ep_s = time.time() - ep_t0
        epoch_times.append(ep_s)
        avg_ep_s = (
            sum(epoch_times[1:]) / len(epoch_times[1:])
            if len(epoch_times) > 1
            else epoch_times[0]
        )
        eta_s = avg_ep_s * (args.epochs - epoch)
        best_tag = " *best" if is_best else ""
        skip_tag = f" | skipped={skipped_steps}" if skipped_steps else ""

        if variational:
            log(
                f"[{name}] ep {epoch:02d}/{args.epochs:02d} | "
                f"train_ce={train_recon_ce:.4f} | val_ce={val['recon_ce']:.4f} | "
                f"val_acc={val['char_acc'] * 100:5.2f}% | "
                f"kl={val['kl']:6.2f} (b={beta_now:.2f}) | elbo={val['elbo_mc']:.2f} | "
                f"{format_duration(ep_s)} | ETA {format_duration(eta_s)}{skip_tag}{best_tag}"
            )
        else:
            log(
                f"[{name}] ep {epoch:02d}/{args.epochs:02d} | "
                f"train_ce={train_recon_ce:.4f} | val_ce={val['recon_ce']:.4f} | "
                f"val_acc={val['char_acc'] * 100:5.2f}% | lr={lr_now:.2e} | "
                f"{format_duration(ep_s)} | ETA {format_duration(eta_s)}{skip_tag}{best_tag}"
            )

    if writer is not None:
        writer.close()

    if best_state is None or best is None:
        raise RuntimeError(f"No se obtuvo ningún checkpoint válido para {name}")

    model.load_state_dict(best_state)
    model.eval()
    latent = batched(lambda xb: model.encode(xb)[0], data["x_sil"], device).numpy()
    sil_cos = float(silhouette_score(latent, data["y_sil"].numpy(), metric="cosine"))

    elapsed_total = time.time() - t0
    log(
        f"[{name}] FIN | best ep {best['epoch']:02d} | val_ce={best['val_recon_ce']:.4f} "
        f"| val_acc={best['val_char_acc'] * 100:.2f}% | kl={best['val_kl']:.2f} "
        f"| sil(cos)={sil_cos:.4f} | total={format_duration(elapsed_total)}"
    )

    summary = {
        "run": name,
        "model": "VAE" if variational else "AE",
        "z_dim": z_dim,
        "beta": beta if variational else None,
        "train_data": data["label"],
        "best_epoch": best["epoch"],
        "val_recon_ce": best["val_recon_ce"],
        "val_recon_ce_mc": best["val_recon_ce_mc"],
        "val_char_acc": best["val_char_acc"],
        "val_kl": best["val_kl"],
        "val_elbo_mc": best["val_elbo_mc"],
        "val_silhouette_cosine": sil_cos,
    }
    payload = {
        "state_dict": cpu_state_dict(model),
        "metadata": {
            "hparams": model.hparams,
            "beta": beta,
            "free_bits": args.free_bits if variational else 0.0,
            "run": name,
            "summary": summary,
            "history": history,
        },
    }
    return summary, history, payload


def plot_curves(histories: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, h in histories.items():
        df = pd.DataFrame(h)
        style = "--" if name.startswith("vae") else "-"
        axes[0].plot(df["epoch"], df["train_recon_ce"], style, label=name)
        axes[1].plot(df["epoch"], df["val_char_acc"], style, label=name)
        if name.startswith("vae"):
            axes[2].plot(df["epoch"], df["val_kl"], style, label=name)
    axes[0].set(title="Train reconstruction CE / char", xlabel="epoch")
    axes[1].set(title="Validation char accuracy", xlabel="epoch")
    axes[2].set(title="VAE: validation KL (no collapse)", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento AE/VAE secuenciales (W4 Fase 4).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--free-bits", type=float, default=DEFAULT_FREE_BITS)
    parser.add_argument("--kl-cycles", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=400_000)
    parser.add_argument("--max-val", type=int, default=40_000)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES)
    parser.add_argument("--fast-verify", action="store_true")
    parser.add_argument(
        "--verbose-rocm",
        action="store_true",
        help="Muestra los logs nativos de C++/MIOpen/Composable Kernel en ROCm.",
    )
    args = parser.parse_args()

    if args.fast_verify:
        args.max_train = min(args.max_train, 60_000)
        args.max_val = min(args.max_val, 10_000)

    silence_rocm = is_rocm_build() and not args.verbose_rocm
    with silence_native_rocm_logs(enabled=silence_rocm):
        t_all = time.time()
        seed_everything(args.seed, deterministic=True)
        device, desc = resolve_device(args.device)
        profile = get_hardware_profile(device, args.profile)
        eff_bs = args.batch_size or profile.batch_size
        log(f"Device: {device} | {desc}")
        log(
            f"Config: profile={profile.name} | batch_size={eff_bs} | epochs={args.epochs} "
            f"| lr={args.lr:g} | max_train={args.max_train:,} | max_val={args.max_val:,} "
            f"| mode={'fast-verify' if args.fast_verify else 'full (6 modelos)'}"
        )
        if silence_rocm:
            log("Logs nativos ROCm/MIOpen silenciados (usa --verbose-rocm para verlos).")

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        t_data = time.time()
        log("Cargando vocabulario y splits codificados...")
        vocab = read_vocab()
        x_train, _ = load_split("train", vocab=vocab, max_samples=args.max_train)
        x_val, y_val = load_split("val", vocab=vocab, max_samples=args.max_val)
        sil = top5_sample(y_val)
        all_data = {
            "x_train": x_train,
            "x_val": x_val,
            "label": "all_clean",
            "x_sil": x_val[sil],
            "y_sil": y_val[sil],
        }
        log(
            f"Datos listos en {format_duration(time.time() - t_data)} | "
            f"train={len(x_train):,} | val={len(x_val):,} | vocab={int(vocab['size'])}"
        )

        if args.fast_verify:
            summary, _, _ = train_run(
                f"vae_fast_b{args.beta:g}",
                128,
                True,
                args.beta,
                all_data,
                int(vocab["size"]),
                device,
                args,
                run_idx=1,
                total_runs=1,
            )
            penalty = 0.0 if summary["val_kl"] >= 15.0 else 0.5
            score = summary["val_char_acc"] - penalty
            print(f"VAL_VAE_SCORE={score:.6f}", flush=True)
            print(f"VAL_KL={summary['val_kl']:.6f}", flush=True)
            return

        summaries: list[dict[str, Any]] = []
        histories: dict[str, list[dict[str, Any]]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        total_runs = 6
        run_counter = 0

        def run(name: str, z: int, variational: bool, beta: float) -> None:
            nonlocal run_counter
            run_counter += 1
            s, h, p = train_run(
                name,
                z,
                variational,
                beta,
                all_data,
                int(vocab["size"]),
                device,
                args,
                run_idx=run_counter,
                total_runs=total_runs,
            )
            summaries.append(s)
            histories[name] = h
            payloads[name] = p

        for z in (32, 64, 128):
            run(f"ae_z{z}", z, False, 0.0)
        best_ae = max((s for s in summaries if s["model"] == "AE"), key=lambda s: s["val_char_acc"])
        z_best = int(best_ae["z_dim"])
        log(
            f"Mejor AE seleccionado: {best_ae['run']} (z_dim={z_best}, "
            f"val_acc={best_ae['val_char_acc'] * 100:.2f}%) -> iniciando VAEs..."
        )

        for beta in (0.5, 1.0, 4.0):
            run(f"vae_b{beta:g}", z_best, True, beta)

        healthy_vaes = [s for s in summaries if s["model"] == "VAE" and s["val_kl"] >= MIN_KL_HEALTHY]
        vae_pool = healthy_vaes if healthy_vaes else [s for s in summaries if s["model"] == "VAE"]
        best_vae = min(vae_pool, key=lambda s: s["val_elbo_mc"])

        save_checkpoint(
            CHECKPOINTS_DIR / "ae_best.pt",
            payloads[best_ae["run"]]["state_dict"],
            metadata=payloads[best_ae["run"]]["metadata"],
        )
        save_checkpoint(
            CHECKPOINTS_DIR / "vae_best.pt",
            payloads[best_vae["run"]]["state_dict"],
            metadata=payloads[best_vae["run"]]["metadata"],
        )
        save_checkpoint(
            CHECKPOINTS_DIR / "ae_normal.pt",
            payloads[best_ae["run"]]["state_dict"],
            metadata=payloads[best_ae["run"]]["metadata"],
        )
        save_checkpoint(
            CHECKPOINTS_DIR / "vae_normal.pt",
            payloads[best_vae["run"]]["state_dict"],
            metadata=payloads[best_vae["run"]]["metadata"],
        )

        table = pd.DataFrame(summaries)
        table.to_csv(REPORTS_DIR / "training_summary.csv", index=False)
        plot_curves(histories)

        log(
            f"Checkpoints guardados -> ae_best={to_relative_path(CHECKPOINTS_DIR / 'ae_best.pt')} "
            f"({best_ae['run']}), vae_best={to_relative_path(CHECKPOINTS_DIR / 'vae_best.pt')} "
            f"({best_vae['run']})"
        )
        log(f"Resumen guardado en {to_relative_path(REPORTS_DIR / 'training_summary.csv')}")
        log(f"Entrenamiento completo finalizado en {format_duration(time.time() - t_all)}")
        print("\nResumen final de runs:", flush=True)
        cols = ["run", "best_epoch", "val_recon_ce", "val_char_acc", "val_kl", "val_silhouette_cosine"]
        print(table[cols].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
