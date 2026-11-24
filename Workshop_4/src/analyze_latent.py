"""Análisis del espacio latente con Silhouette coseno + Linear Probing (W4 — Fase 4)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared.config import SEED, to_relative_path  # noqa: E402
from shared.data import load_split_df, read_vocab  # noqa: E402
from shared.device import DEVICE_CHOICES, is_rocm_build, resolve_device, seed_everything  # noqa: E402
from Workshop_4.src.autoencoder import (  # noqa: E402
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    batched,
    load_model,
    load_split,
    load_w2_transformer,
)
from Workshop_4.src.train_ae import top5_sample  # noqa: E402


def extract_intrinsic_labels(split: str, indices: np.ndarray) -> dict[str, np.ndarray]:
    df = load_split_df(split, columns=["character_id", "x"]).iloc[indices]
    has_strong_punct = df["x"].str.contains(r"[!?;:]", regex=True).astype(int).to_numpy()
    high_upper = (df["x"].str.count(r"[A-Z]") >= 4).astype(int).to_numpy()
    return {"has_strong_punct": has_strong_punct, "high_uppercase": high_upper}


def linear_probe(
    z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(z_train)
    x_te = scaler.transform(z_test)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    clf.fit(x_tr, y_train)
    preds = clf.predict(x_te)
    return (
        float(accuracy_score(y_test, preds)),
        float(f1_score(y_test, preds, average="macro", zero_division=0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza el espacio latente de W4 (Fase 4).")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--n-samples", type=int, default=3000)
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

        log(f"Cargando splits val/test y propiedades intrinsecas (n_samples={args.n_samples:,})...")
        vocab = read_vocab()
        itos = vocab["itos"]
        x_val, y_val = load_split("val", vocab=vocab, max_samples=20_000)
        x_test, y_test = load_split("test", vocab=vocab, max_samples=20_000)

        val_idx = top5_sample(y_val, n=args.n_samples)
        test_idx = top5_sample(y_test, n=args.n_samples)

        xv, yv = x_val[val_idx], y_val[val_idx].numpy()
        xt, yt = x_test[test_idx], y_test[test_idx].numpy()

        val_orig_idx = np.sort(
            np.random.default_rng(SEED).choice(len(load_split_df("val", columns=["y_id"])), 20_000, replace=False)
        )[val_idx]
        test_orig_idx = np.sort(
            np.random.default_rng(SEED).choice(len(load_split_df("test", columns=["y_id"])), 20_000, replace=False)
        )[test_idx]
        val_props = extract_intrinsic_labels("val", val_orig_idx)
        test_props = extract_intrinsic_labels("test", test_orig_idx)

        log("Cargando checkpoints ae_best.pt, vae_best.pt y Transformer W2...")
        ae, ae_meta = load_model(CHECKPOINTS_DIR / "ae_best.pt", device)
        vae, vae_meta = load_model(CHECKPOINTS_DIR / "vae_best.pt", device)
        w2 = load_w2_transformer(device)

        encoders = {
            f"AE ({ae_meta.get('run', 'ae')})": lambda xb: ae.encode(xb)[0],
            f"VAE ({vae_meta.get('run', 'vae')}, mu)": lambda xb: vae.encode(xb)[0],
            "Transformer W2 (supervised)": lambda xb: w2.features(xb.long()),
        }

        rows = []
        fig, axes = plt.subplots(len(encoders), 1, figsize=(8, 4.2 * len(encoders)))
        labels = sorted(set(yt.tolist()))
        total_enc = len(encoders)

        for idx_e, (ax, (name, fn)) in enumerate(zip(axes, encoders.items(), strict=True), start=1):
            t_step = time.time()
            log(f"[{idx_e}/{total_enc}] Analizando representacion: {name} (embeddings + probes + t-SNE)...")
            zv = batched(fn, xv, device).numpy()
            zt = batched(fn, xt, device).numpy()

            sil_euc = float(silhouette_score(zt, yt, metric="euclidean"))
            sil_cos = float(silhouette_score(zt, yt, metric="cosine"))
            _, f1_next = linear_probe(zv, yv, zt, yt)
            _, f1_punct = linear_probe(
                zv, val_props["has_strong_punct"], zt, test_props["has_strong_punct"]
            )
            _, f1_upper = linear_probe(
                zv, val_props["high_uppercase"], zt, test_props["high_uppercase"]
            )

            rows.append(
                {
                    "representation": name,
                    "dimensions": zt.shape[1],
                    "silhouette_euclidean": sil_euc,
                    "silhouette_cosine": sil_cos,
                    "probe_next_char_f1": f1_next,
                    "probe_strong_punct_f1": f1_punct,
                    "probe_high_upper_f1": f1_upper,
                }
            )

            proj = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(zt[:1500])
            yt_sub = yt[:1500]
            for lbl in labels:
                m = yt_sub == lbl
                ch = "<space>" if itos[lbl] == " " else itos[lbl]
                ax.scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.6, label=ch)
            ax.set_title(
                f"{name} | sil(cos)={sil_cos:.3f} | probe punct F1={f1_punct:.3f} | next F1={f1_next:.3f}",
                fontsize=9,
            )
            ax.legend(title="next char", markerscale=2.5, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            log(
                f"  sil(cos)={sil_cos:.4f} | next_f1={f1_next:.4f} | "
                f"punct_f1={f1_punct:.4f} | upper_f1={f1_upper:.4f} "
                f"({format_duration(time.time() - t_step)})"
            )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "latent_space.png", dpi=110)
        plt.close(fig)

        df = pd.DataFrame(rows)
        df.to_csv(REPORTS_DIR / "silhouette_scores.csv", index=False)
        print("\nAnalisis latente y Linear Probing:", flush=True)
        print(df.round(4).to_string(index=False), flush=True)
        log(
            f"Figura y tabla guardadas en {to_relative_path(REPORTS_DIR)} | "
            f"total={format_duration(time.time() - t_all)}"
        )


if __name__ == "__main__":
    main()
