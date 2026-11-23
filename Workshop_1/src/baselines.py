"""Baselines clásicos para Workshop 1 encapsulados en sklearn.pipeline.Pipeline."""

from __future__ import annotations

import argparse
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

for _p in (REPO_ROOT, PROJECT_ROOT, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.config import SEED, to_relative_path
from shared.data import load_split_df, read_vocab
from shared.device import seed_everything
from shared.metrics import compute_multiclass_metrics, slice_metrics_by_char_type

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_NGRAM_WINDOW = 20
DEFAULT_POS_WINDOW = 5
DEFAULT_MAX_TRAIN_ROWS = 300_000


class TailSubstringExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window: int = DEFAULT_NGRAM_WINDOW) -> None:
        self.window = window

    def fit(self, X: Any, y: Any = None) -> TailSubstringExtractor:
        return self

    def transform(self, X: Any) -> list[str]:
        return [str(text)[-self.window :] for text in X]


class PositionalCharExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pos_window: int = DEFAULT_POS_WINDOW) -> None:
        self.pos_window = pos_window

    def fit(self, X: Any, y: Any = None) -> PositionalCharExtractor:
        return self

    def transform(self, X: Any) -> np.ndarray:
        rows = []
        for text in X:
            s = str(text)[-self.pos_window :].rjust(self.pos_window)
            rows.append(list(s))
        return np.asarray(rows, dtype=object)


class MorphologicalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window: int = DEFAULT_NGRAM_WINDOW) -> None:
        self.window = window

    def fit(self, X: Any, y: Any = None) -> MorphologicalFeatureExtractor:
        return self

    def transform(self, X: Any) -> sparse.csr_matrix:
        n = len(X)
        feats = np.zeros((n, 8), dtype=np.float32)
        punct_set = set(string.punctuation)
        upper_set = set(string.ascii_uppercase)
        lower_set = set(string.ascii_lowercase)
        word_chars = upper_set | lower_set | {"'"}

        for i, text in enumerate(X):
            s = str(text)[-self.window :]
            if not s:
                continue
            last = s[-1]
            prev = s[-2] if len(s) >= 2 else " "
            feats[i, 0] = 1.0 if last == " " else 0.0
            feats[i, 1] = 1.0 if last in lower_set else 0.0
            feats[i, 2] = 1.0 if last in upper_set else 0.0
            feats[i, 3] = 1.0 if last in punct_set else 0.0
            feats[i, 4] = 1.0 if prev == " " else 0.0
            feats[i, 5] = 1.0 if prev in punct_set else 0.0

            word_len = 0
            for ch in reversed(s):
                if ch in word_chars:
                    word_len += 1
                else:
                    break
            feats[i, 6] = min(word_len / 10.0, 1.0)
            feats[i, 7] = sum(1 for ch in s if ch in upper_set) / float(len(s))

        return sparse.csr_matrix(feats)


def build_feature_pipeline(
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    pos_window: int = DEFAULT_POS_WINDOW,
    max_ngram_features: int = 50_000,
    min_df: int = 3,
) -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "char_ngrams",
                Pipeline(
                    [
                        ("tail", TailSubstringExtractor(window=ngram_window)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(1, 5),
                                sublinear_tf=True,
                                lowercase=False,
                                min_df=min_df,
                                max_features=max_ngram_features,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "pos_onehot",
                Pipeline(
                    [
                        ("pos", PositionalCharExtractor(pos_window=pos_window)),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", dtype=np.float32)),
                    ]
                ),
            ),
            ("morph", MorphologicalFeatureExtractor(window=ngram_window)),
        ]
    )


def build_baseline_pipelines(
    *,
    ngram_window: int = DEFAULT_NGRAM_WINDOW,
    pos_window: int = DEFAULT_POS_WINDOW,
    max_ngram_features: int = 50_000,
    min_df: int = 3,
    seed: int = SEED,
) -> dict[str, Pipeline]:
    common_kwargs = {
        "ngram_window": ngram_window,
        "pos_window": pos_window,
        "max_ngram_features": max_ngram_features,
        "min_df": min_df,
    }
    return {
        "logistic_sgd_balanced": Pipeline(
            [
                ("features", build_feature_pipeline(**common_kwargs)),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=1e-5,
                        class_weight="balanced",
                        max_iter=40,
                        early_stopping=False,
                        tol=1e-3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "complement_nb": Pipeline(
            [
                ("features", build_feature_pipeline(**common_kwargs)),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
        "linear_huber_balanced": Pipeline(
            [
                ("features", build_feature_pipeline(**common_kwargs)),
                (
                    "clf",
                    SGDClassifier(
                        loss="modified_huber",
                        penalty="l2",
                        alpha=1e-5,
                        class_weight="balanced",
                        max_iter=40,
                        early_stopping=False,
                        tol=1e-3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def stratified_sample_df(
    df: pd.DataFrame,
    n_samples: int,
    target_col: str = "y_id",
    seed: int = SEED,
) -> pd.DataFrame:
    if n_samples <= 0 or n_samples >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    counts = df[target_col].value_counts()
    rare_classes = counts[counts < 5].index
    rare_mask = df[target_col].isin(rare_classes)
    rare_idx = df.index[rare_mask].to_numpy()

    remaining_budget = max(0, n_samples - len(rare_idx))
    common_df = df.loc[~rare_mask]
    if remaining_budget >= len(common_df):
        chosen = np.concatenate([rare_idx, common_df.index.to_numpy()])
        return df.loc[np.sort(chosen)].reset_index(drop=True)

    selected_parts = [rare_idx]
    common_counts = common_df[target_col].value_counts()
    proportions = common_counts / common_counts.sum()
    allocations = np.maximum(1, np.floor(proportions * remaining_budget).astype(int))
    allocations = np.minimum(allocations, common_counts)

    for cls, k in allocations.items():
        cls_indices = common_df.index[common_df[target_col] == cls].to_numpy()
        selected_parts.append(rng.choice(cls_indices, size=int(k), replace=False))

    chosen = np.sort(np.concatenate(selected_parts))
    return df.loc[chosen].reset_index(drop=True)


def tune_sgd_alpha_group_cv(
    train_df: pd.DataFrame,
    loss: str,
    alphas: tuple[float, ...] = (1e-6, 1e-5, 1e-4),
    n_splits: int = 3,
    cv_rows: int = 60_000,
    seed: int = SEED,
) -> float:
    sub = stratified_sample_df(train_df, min(cv_rows, len(train_df)), seed=seed)
    X_text = sub["x"].tolist()
    y = sub["y_id"].to_numpy()
    groups = sub["character_id"].to_numpy()

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    best_alpha = alphas[0]
    best_score = -1.0

    for alpha in alphas:
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X_text, y, groups):
            pipe = Pipeline(
                [
                    ("features", build_feature_pipeline(max_ngram_features=25_000)),
                    (
                        "clf",
                        SGDClassifier(
                            loss=loss,
                            penalty="l2",
                            alpha=alpha,
                            class_weight="balanced",
                            max_iter=25,
                            early_stopping=False,
                            tol=1e-3,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            X_tr = [X_text[i] for i in tr_idx]
            X_va = [X_text[i] for i in va_idx]
            pipe.fit(X_tr, y[tr_idx])
            preds = pipe.predict(X_va)
            fold_scores.append(float(f1_score(y[va_idx], preds, average="macro", zero_division=0)))
        mean_f1 = float(np.mean(fold_scores))
        if mean_f1 > best_score:
            best_score = mean_f1
            best_alpha = alpha
    return best_alpha


def predict_proba_aligned(
    pipeline: Pipeline,
    texts: list[str],
    vocab_size: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    n = len(texts)
    full_probs = np.zeros((n, vocab_size), dtype=np.float32)
    classes = pipeline.named_steps["clf"].classes_

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_probs = pipeline.predict_proba(texts[start:end])
        full_probs[start:end, classes] = chunk_probs.astype(np.float32)
    return full_probs


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    itos: list[str],
    title: str,
    output_path: Path,
) -> None:
    top20 = pd.Series(y_true).value_counts().index[:20].to_numpy()
    mask = np.isin(y_true, top20)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top20, normalize="true")
    labels = ["<space>" if itos[c] == " " else itos[c] for c in top20]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa los baselines de Workshop 1.")
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--max-val-rows", type=int, default=100_000)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vocab = read_vocab()
    vocab_size = int(vocab["size"])
    itos = vocab["itos"]

    train_full = load_split_df("train", columns=["character_id", "x", "y_id"])
    val_full = load_split_df("val", columns=["character_id", "x", "y_id"])
    test_full = load_split_df("test", columns=["character_id", "x", "y_id"])

    train_df = stratified_sample_df(train_full, args.max_train_rows, seed=args.seed)
    val_df = stratified_sample_df(val_full, args.max_val_rows, seed=args.seed)
    test_10k_idx = test_full.sample(n=min(10_000, len(test_full)), random_state=SEED).index.to_numpy()

    pipelines = build_baseline_pipelines(seed=args.seed)
    if not args.skip_cv:
        best_lr_alpha = tune_sgd_alpha_group_cv(train_df, loss="log_loss", seed=args.seed)
        best_huber_alpha = tune_sgd_alpha_group_cv(train_df, loss="modified_huber", seed=args.seed)
        pipelines["logistic_sgd_balanced"].set_params(clf__alpha=best_lr_alpha)
        pipelines["linear_huber_balanced"].set_params(clf__alpha=best_huber_alpha)
        print(f"GroupKFold best alphas -> log_loss={best_lr_alpha:g}, huber={best_huber_alpha:g}")

    X_train = train_df["x"].tolist()
    y_train = train_df["y_id"].to_numpy()
    X_val = val_df["x"].tolist()
    y_val = val_df["y_id"].to_numpy()
    X_test = test_full["x"].tolist()
    y_test = test_full["y_id"].to_numpy()

    summary_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for name, pipe in pipelines.items():
        print(f"\n[Workshop 1] Entrenando pipeline: {name} ({len(train_df):,} filas)...")
        pipe.fit(X_train, y_train)

        ckpt_path = CHECKPOINTS_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipe, ckpt_path, compress=3)

        val_probs = predict_proba_aligned(pipe, X_val, vocab_size)
        val_m = compute_multiclass_metrics(y_val, val_probs, is_probs=True, max_ranking_samples=30_000)

        test_probs = predict_proba_aligned(pipe, X_test, vocab_size)
        test_m = compute_multiclass_metrics(y_test, test_probs, is_probs=True, max_ranking_samples=50_000)
        test_10k_m = compute_multiclass_metrics(
            y_test[test_10k_idx], test_probs[test_10k_idx], is_probs=True, max_ranking_samples=None
        )

        y_test_pred = test_probs.argmax(axis=1)
        for s in slice_metrics_by_char_type(y_test, y_test_pred, itos):
            slice_rows.append({"model": name, **s})

        save_confusion_matrix_plot(
            y_test,
            y_test_pred,
            itos,
            f"Workshop 1 {name} - Test Top-20 Confusion Matrix",
            run_dir / f"{name}_cm_test.png",
        )

        row = {
            "model": name,
            "train_rows": len(train_df),
            "val_accuracy": val_m["accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "test_accuracy": test_m["accuracy"],
            "test_top5_accuracy": test_m["top5_accuracy"],
            "test_f1_macro": test_m["f1_macro"],
            "test_f1_weighted": test_m["f1_weighted"],
            "test_pr_auc_macro": test_m["pr_auc_macro"],
            "test_auc_roc_ovr_macro": test_m["auc_roc_ovr_macro"],
            "test_bpc": test_m["bpc"],
            "test_10k_accuracy": test_10k_m["accuracy"],
            "test_10k_f1_macro": test_10k_m["f1_macro"],
            "checkpoint": to_relative_path(ckpt_path),
        }
        summary_rows.append(row)
        print(
            f"  {name}: test_acc={test_m['accuracy']:.4f} | test_f1_macro={test_m['f1_macro']:.4f} "
            f"| test_pr_auc={test_m['pr_auc_macro']:.4f} | bpc={test_m['bpc']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "baselines_summary.csv", index=False)
    summary_df.to_csv(CHECKPOINTS_DIR / "baselines_summary.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(REPORTS_DIR / "baselines_slices.csv", index=False)

    print("\nResumen Workshop 1 guardado en Workshop_1/reports/baselines_summary.csv:")
    print(summary_df[["model", "test_accuracy", "test_f1_macro", "test_pr_auc_macro", "test_bpc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
