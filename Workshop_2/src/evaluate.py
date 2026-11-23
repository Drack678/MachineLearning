"""Evaluación estandarizada en TEST del Transformer de Workshop 2 (Fase 2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint
from shared.config import SEED, W1_DIR
from shared.data import load_encoded_split, load_split_df, read_vocab
from shared.device import DEVICE_CHOICES, get_amp_config, resolve_device, seed_everything
from shared.metrics import (
    apply_logit_adjustment,
    compute_multiclass_metrics,
    slice_metrics_by_char_type,
)
from shared.models import CharTransformer
from Workshop_2.src.train import BEST_CHECKPOINT, FIGURES_DIR, REPORTS_DIR, predict_logits


def load_transformer_safe(
    path: Path, device: torch.device
) -> tuple[CharTransformer, dict]:
    state_dict, meta = load_checkpoint(path, device="cpu")
    model = CharTransformer(**meta["model_hparams"])
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta


def show_char(char: str) -> str:
    return "<space>" if char == " " else char


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el Transformer W2 en TEST.")
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    seed_everything(SEED, deterministic=True)
    device, desc = resolve_device(args.device)
    amp_enabled, amp_dtype, _ = get_amp_config(device)
    print(f"Device: {device} | {desc}")

    vocab = read_vocab()
    itos = vocab["itos"]
    vocab_size = int(vocab["size"])

    x_test, y_test = load_encoded_split("test", vocab=vocab)
    y_true = y_test.numpy().astype(np.int64)
    sample_10k = load_split_df("test", columns=["y_id"]).sample(n=10_000, random_state=SEED).index.to_numpy()

    _, y_train = load_encoded_split("train", vocab=vocab)
    train_counts = torch.bincount(y_train.long(), minlength=vocab_size).numpy()

    priors = torch.as_tensor(train_counts, dtype=torch.float64) + 1e-9
    maj_logits = (priors / priors.sum()).log().float().expand(len(y_true), -1)
    rows = [
        {
            "model": "Majority class (space)",
            "split": "test_full",
            **compute_multiclass_metrics(y_true, maj_logits, max_ranking_samples=50_000),
        }
    ]

    w1_summary = W1_DIR / "reports" / "baselines_summary.csv"
    if w1_summary.exists():
        w1_df = pd.read_csv(w1_summary)
        for r in w1_df.itertuples():
            rows.append(
                {
                    "model": f"W1 {r.model}",
                    "split": "test_full",
                    "accuracy": r.test_accuracy,
                    "top5_accuracy": r.test_top5_accuracy,
                    "f1_macro": r.test_f1_macro,
                    "f1_weighted": r.test_f1_weighted,
                    "pr_auc_macro": r.test_pr_auc_macro,
                    "auc_roc_ovr_macro": r.test_auc_roc_ovr_macro,
                    "bpc": r.test_bpc,
                }
            )

    model, meta = load_transformer_safe(args.checkpoint, device)
    test_logits = predict_logits(
        model, x_test, device, batch_size=args.batch_size, amp_enabled=amp_enabled, amp_dtype=amp_dtype
    )
    tau = float(meta.get("logit_adj_tau", 0.0))
    counts_for_adj = np.asarray(meta.get("train_class_counts", train_counts))
    test_logits_adj = apply_logit_adjustment(test_logits, counts_for_adj, tau)

    m_cal_full = compute_multiclass_metrics(y_true, test_logits, max_ranking_samples=50_000)
    m_adj_full = compute_multiclass_metrics(y_true, test_logits_adj, max_ranking_samples=50_000)
    m_cal_10k = compute_multiclass_metrics(y_true[sample_10k], test_logits[sample_10k], max_ranking_samples=None)
    m_adj_10k = compute_multiclass_metrics(y_true[sample_10k], test_logits_adj[sample_10k], max_ranking_samples=None)

    rows.append({"model": "Transformer W2 (calibrated, tau=0)", "split": "test_full", **m_cal_full})
    rows.append({"model": f"Transformer W2 (logit-adj, tau={tau:.2f})", "split": "test_full", **m_adj_full})
    rows.append({"model": "Transformer W2 (calibrated, tau=0)", "split": "test_10k", **m_cal_10k})
    rows.append({"model": f"Transformer W2 (logit-adj, tau={tau:.2f})", "split": "test_10k", **m_adj_10k})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "results_comparison.csv", index=False)
    print("\nResultados comparativos (Workshop_2/reports/results_comparison.csv):")
    cols = ["model", "split", "accuracy", "top5_accuracy", "f1_macro", "pr_auc_macro", "bpc", "ece"]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].round(4).to_string(index=False))

    y_pred_cal = test_logits.argmax(dim=1).numpy()
    y_pred_adj = test_logits_adj.argmax(dim=1).numpy()
    slices_df = pd.DataFrame(
        [{"mode": "calibrated", **s} for s in slice_metrics_by_char_type(y_true, y_pred_cal, itos)]
        + [{"mode": f"logit_adj_tau{tau:.2f}", **s} for s in slice_metrics_by_char_type(y_true, y_pred_adj, itos)]
    )
    slices_df.to_csv(REPORTS_DIR / "test_slices.csv", index=False)

    top20 = pd.Series(y_true).value_counts().index[:20].to_numpy()
    mask = np.isin(y_true, top20)
    cm = confusion_matrix(y_true[mask], y_pred_cal[mask], labels=top20, normalize="true")
    labels = [show_char(itos[c]) for c in top20]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(20), labels, rotation=90)
    ax.set_yticks(range(20), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix on test (top-20 characters, calibrated)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_top20.png", dpi=110)
    plt.close(fig)

    classes = np.unique(y_true)
    per_class = pd.DataFrame(
        {
            "char": [show_char(itos[c]) for c in classes],
            "support": np.bincount(y_true)[classes],
            "recall_cal": recall_score(y_true, y_pred_cal, labels=classes, average=None, zero_division=0),
            "recall_adj": recall_score(y_true, y_pred_adj, labels=classes, average=None, zero_division=0),
        }
    )
    per_class = per_class[per_class["support"] >= 100]
    print("\nPeores 10 clases (soporte >= 100):")
    print(per_class.nsmallest(10, "recall_cal").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
