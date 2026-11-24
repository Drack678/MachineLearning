"""Evaluación en TEST de reconstrucción y Detección de Anomalías reformulada (W4 — Fase 4)."""

from __future__ import annotations

import argparse
import string
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shared.config import SEED, to_relative_path  # noqa: E402
from shared.data import read_vocab  # noqa: E402
from shared.device import DEVICE_CHOICES, is_rocm_build, resolve_device, seed_everything  # noqa: E402
from Workshop_4.src.autoencoder import (  # noqa: E402
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    batched,
    decode,
    kl_divergence,
    load_model,
    load_split,
    load_w2_transformer,
    reconstruction_loss,
)

N_EXAMPLES = 8


def char_type(c: str) -> str:
    if c == " ":
        return "space"
    if c in string.ascii_lowercase:
        return "lowercase"
    if c in string.ascii_uppercase:
        return "uppercase"
    return "punctuation/other"


def build_anomaly_benchmark(
    x_clean: torch.Tensor,
    vocab: dict[str, Any],
    anomaly_ratio: float = 0.05,
    seed: int = SEED,
) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = len(x_clean)
    n_anom = max(2, int(round(n_total * anomaly_ratio)))
    n_corrupt = n_anom // 2
    n_ood = n_anom - n_corrupt
    n_normal = n_total - n_anom

    perm = rng.permutation(n_total)
    normal_x = x_clean[torch.from_numpy(perm[:n_normal])].clone()

    corrupt_base = x_clean[torch.from_numpy(perm[n_normal : n_normal + n_corrupt])].clone()
    vocab_size = int(vocab["size"])
    seq_len = x_clean.size(1)
    mask = torch.from_numpy(rng.random((n_corrupt, seq_len)) < 0.35)
    rand_chars = torch.from_numpy(rng.integers(1, vocab_size, size=(n_corrupt, seq_len), dtype=np.int64))
    corrupt_base[mask] = rand_chars[mask].to(corrupt_base.dtype)

    stoi = vocab["stoi"]
    ood_pool = [
        stoi[c]
        for c in "0123456789[]():;!?-\"&ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if c in stoi
    ]
    ood_arr = rng.choice(ood_pool, size=(n_ood, seq_len)).astype(np.int64)
    ood_x = torch.from_numpy(ood_arr).to(x_clean.dtype)

    x_mix = torch.cat([normal_x, corrupt_base, ood_x], dim=0)
    y_anom = np.concatenate(
        [np.zeros(n_normal, dtype=np.int32), np.ones(n_anom, dtype=np.int32)]
    )
    return x_mix, y_anom


def calibrate_threshold_on_val(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    if len(thr) == 0:
        return float(np.median(val_scores))
    f1s = 2.0 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_idx = int(np.argmax(f1s))
    return float(thr[best_idx])


@torch.inference_mode()
def reconstruct(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batched(lambda xb: model(xb, sample_posterior=False)[0].softmax(-1), x, device)


def reconstruction_metrics(probs: torch.Tensor, x: torch.Tensor, itos: list[str]):
    x_long = x.long()
    onehot = torch.nn.functional.one_hot(x_long, probs.size(-1)).float()
    pred = probs.argmax(-1)
    correct = (pred == x_long).float()
    metrics: dict[str, float] = {
        "mse": float(((onehot - probs) ** 2).sum(-1).mean().item()),
        "ce_per_char": float(-torch.log(probs.gather(-1, x_long.unsqueeze(-1)).clamp_min(1e-12)).mean().item()),
        "char_accuracy": float(correct.mean().item()),
    }
    types = np.array([char_type(c) for c in itos])[x_long.numpy()]
    for t in ("space", "lowercase", "uppercase", "punctuation/other"):
        metrics[f"acc_{t}"] = float(correct.numpy()[types == t].mean())
    return metrics, correct


def transformer_window_nll(w2, x: torch.Tensor, device: torch.device) -> np.ndarray:
    def _score(xb: torch.Tensor) -> torch.Tensor:
        xb_l = xb.long()
        logits = w2(xb_l[:, :-1], return_all_tokens=True)
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), xb_l[:, 1:], reduction="none")
        return ce.mean(dim=1)

    return batched(_score, x, device, batch_size=512).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa AE/VAE en TEST (Workshop 4 Fase 4).")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument(
        "--verbose-rocm",
        action="store_true",
        help="Muestra los logs nativos de C++/MIOpen/Composable Kernel en ROCm.",
    )
    args = parser.parse_args()

    silence_rocm = is_rocm_build() and not args.verbose_rocm
    with silence_native_rocm_logs(enabled=silence_rocm):
        t_all = time.time()
        seed_everything(SEED)
        device, desc = resolve_device(args.device)
        log(f"Device: {device} | {desc}")
        if silence_rocm:
            log("Logs nativos ROCm/MIOpen silenciados (usa --verbose-rocm para verlos).")

        log("Cargando vocabulario y splits val/test...")
        vocab = read_vocab()
        itos = vocab["itos"]
        x_val, _ = load_split("val", vocab=vocab, max_samples=20_000)
        x_test, _ = load_split("test", vocab=vocab, max_samples=20_000)
        log(f"Splits listos | val={len(x_val):,} | test={len(x_test):,}")

        log("Cargando checkpoints ae_best.pt y vae_best.pt...")
        ae, ae_meta = load_model(CHECKPOINTS_DIR / "ae_best.pt", device)
        vae, vae_meta = load_model(CHECKPOINTS_DIR / "vae_best.pt", device)

        rows = []
        correct = {}
        probs = {}
        for name, model, meta in (("AE", ae, ae_meta), ("VAE", vae, vae_meta)):
            t_step = time.time()
            log(f"Evaluando reconstruccion en TEST para {name} ({meta.get('run', name)})...")
            probs[name] = reconstruct(model, x_test, device)
            m, correct[name] = reconstruction_metrics(probs[name], x_test, itos)
            rows.append({"model": f"{name} ({meta.get('run', name)})", **m})
            log(
                f"  {name}: char_acc={m['char_accuracy'] * 100:.2f}% | "
                f"ce={m['ce_per_char']:.4f} | mse={m['mse']:.4f} "
                f"({format_duration(time.time() - t_step)})"
            )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(REPORTS_DIR / "reconstruction_metrics.csv", index=False)
        print("\nReconstruccion en TEST:", flush=True)
        print(table.round(4).to_string(index=False), flush=True)

        idx = np.random.default_rng(SEED).choice(len(x_test), N_EXAMPLES, replace=False)
        lines = []
        for i in idx:
            for label, seq in (
                ("original", x_test[i].tolist()),
                ("AE", probs["AE"][i].argmax(-1).tolist()),
                ("VAE", probs["VAE"][i].argmax(-1).tolist()),
            ):
                lines.append(f"{label:9s}| {decode(seq, itos)}")
            lines.append("")
        (REPORTS_DIR / "reconstruction_examples.txt").write_text("\n".join(lines), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        for name in ("AE", "VAE"):
            ax.plot(range(1, x_test.size(1) + 1), correct[name].mean(0).numpy(), label=name)
        ax.set(title="Reconstruction accuracy per position (test)", xlabel="position", ylabel="char accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "accuracy_per_position.png", dpi=110)
        plt.close(fig)

        log(f"Construyendo benchmark de anomalias (ratio={args.anomaly_ratio:.1%})...")
        x_val_anom, y_val_anom = build_anomaly_benchmark(x_val, vocab, anomaly_ratio=args.anomaly_ratio, seed=SEED)
        x_test_anom, y_test_anom = build_anomaly_benchmark(x_test, vocab, anomaly_ratio=args.anomaly_ratio, seed=SEED + 1)

        log("Cargando ae_normal.pt, vae_normal.pt y Transformer W2...")
        ae_n, _ = load_model(CHECKPOINTS_DIR / "ae_normal.pt", device)
        vae_n, _ = load_model(CHECKPOINTS_DIR / "vae_normal.pt", device)
        w2 = load_w2_transformer(device)

        def _ae_err(model, x_in):
            return batched(lambda xb: reconstruction_loss(model(xb, sample_posterior=False)[0], xb), x_in, device).numpy()

        def _vae_elbo(model, x_in):
            def _fn(xb):
                logits, mu, lv = model(xb, sample_posterior=False)
                return reconstruction_loss(logits, xb) + kl_divergence(mu, lv)
            return batched(_fn, x_in, device).numpy()

        scorers = {
            "AE sequential (recon error)": lambda x_in: _ae_err(ae_n, x_in),
            "VAE sequential (recon error)": lambda x_in: _ae_err(vae_n, x_in),
            "VAE sequential (recon + KL)": lambda x_in: _vae_elbo(vae_n, x_in),
            "Transformer W2 (window NLL)": lambda x_in: transformer_window_nll(w2, x_in, device),
        }

        anom_rows = []
        total_scorers = len(scorers)
        for idx_s, (method, fn) in enumerate(scorers.items(), start=1):
            t_step = time.time()
            log(f"[{idx_s}/{total_scorers}] Evaluando detector: {method}...")
            val_s = fn(x_val_anom)
            thr = calibrate_threshold_on_val(y_val_anom, val_s)
            test_s = fn(x_test_anom)
            y_hat = (test_s >= thr).astype(np.int32)
            row = {
                "method": method,
                "pr_auc": float(average_precision_score(y_test_anom, test_s)),
                "roc_auc": float(roc_auc_score(y_test_anom, test_s)),
                "f1": float(f1_score(y_test_anom, y_hat, zero_division=0)),
                "precision": float(precision_score(y_test_anom, y_hat, zero_division=0)),
                "recall": float(recall_score(y_test_anom, y_hat, zero_division=0)),
                "val_threshold": thr,
            }
            anom_rows.append(row)
            log(
                f"  pr_auc={row['pr_auc']:.4f} | roc_auc={row['roc_auc']:.4f} "
                f"| f1={row['f1']:.4f} ({format_duration(time.time() - t_step)})"
            )

        anom_df = pd.DataFrame(anom_rows)
        anom_df.to_csv(REPORTS_DIR / "anomaly_detection_metrics.csv", index=False)
        print(f"\nDeteccion de anomalias en TEST ({y_test_anom.mean():.1%} anomalias):", flush=True)
        print(anom_df.round(4).to_string(index=False), flush=True)
        log(
            f"Reportes guardados en {to_relative_path(REPORTS_DIR)} | "
            f"total={format_duration(time.time() - t_all)}"
        )


if __name__ == "__main__":
    main()
