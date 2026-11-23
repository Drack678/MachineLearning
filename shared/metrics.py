"""Métricas estandarizadas para desbalanceo extremo, slices, calibración y ajuste de logits."""

from __future__ import annotations

import math
import string
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def char_category(char: str) -> str:
    if char == " ":
        return "space"
    if char in string.ascii_lowercase:
        return "lowercase"
    if char in string.ascii_uppercase:
        return "uppercase"
    if char == "<UNK>":
        return "unk"
    return "punctuation/other"


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count > 0:
            ece += (count / n) * abs(float(accuracies[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def compute_pr_auc_macro(y_true: np.ndarray, probs: np.ndarray) -> float:
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    scores = []
    for cls in present:
        y_bin = (y_true == cls).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        scores.append(float(average_precision_score(y_bin, probs[:, cls])))
    return float(np.mean(scores)) if scores else float("nan")


def compute_roc_auc_ovr_macro(y_true: np.ndarray, probs: np.ndarray) -> float:
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    scores = []
    for cls in present:
        y_bin = (y_true == cls).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        scores.append(float(roc_auc_score(y_bin, probs[:, cls])))
    return float(np.mean(scores)) if scores else float("nan")


def compute_multiclass_metrics(
    y_true: np.ndarray,
    logits_or_probs: torch.Tensor | np.ndarray,
    *,
    is_probs: bool = False,
    with_ranking_metrics: bool = True,
    max_ranking_samples: int | None = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    if isinstance(logits_or_probs, torch.Tensor):
        tensor = logits_or_probs.detach().cpu().float()
    else:
        tensor = torch.as_tensor(logits_or_probs, dtype=torch.float32)

    if is_probs:
        probs_t = tensor.clamp_min(1e-12)
        probs_t = probs_t / probs_t.sum(dim=-1, keepdim=True)
        nll = float(-torch.log(probs_t.gather(1, torch.from_numpy(y_true_np).unsqueeze(1))).mean().item())
    else:
        nll = float(torch.nn.functional.cross_entropy(tensor, torch.from_numpy(y_true_np)).item())
        probs_t = torch.softmax(tensor, dim=-1)

    bpc = nll / math.log(2.0)
    pred = probs_t.argmax(dim=1).numpy()

    k = min(5, probs_t.size(1))
    top5_idx = probs_t.topk(k, dim=1).indices.numpy()
    top5_acc = float(np.any(top5_idx == y_true_np[:, None], axis=1).mean())

    metrics: dict[str, float] = {
        "accuracy": float((pred == y_true_np).mean()),
        "top5_accuracy": top5_acc,
        "precision_macro": float(precision_score(y_true_np, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_np, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_np, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_np, pred, average="weighted", zero_division=0)),
        "cross_entropy_nats": nll,
        "bpc": bpc,
        "ece": compute_ece(y_true_np, probs_t.numpy()),
    }

    if with_ranking_metrics:
        probs_np = probs_t.numpy()
        if max_ranking_samples is not None and len(y_true_np) > max_ranking_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y_true_np), size=max_ranking_samples, replace=False)
            y_sub, p_sub = y_true_np[idx], probs_np[idx]
        else:
            y_sub, p_sub = y_true_np, probs_np
        metrics["pr_auc_macro"] = compute_pr_auc_macro(y_sub, p_sub)
        metrics["auc_roc_ovr_macro"] = compute_roc_auc_ovr_macro(y_sub, p_sub)

    return metrics


def slice_metrics_by_char_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    itos: list[str],
    rare_threshold: int = 100,
) -> list[dict[str, Any]]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cats = np.array([char_category(c) for c in itos])[y_true_np]
    counts = np.bincount(y_true_np, minlength=len(itos))
    rare_mask = counts[y_true_np] < rare_threshold

    slices: list[dict[str, Any]] = []
    for cat in ("space", "lowercase", "uppercase", "punctuation/other"):
        mask = cats == cat
        if int(mask.sum()) == 0:
            continue
        slices.append(
            {
                "slice": cat,
                "support": int(mask.sum()),
                "accuracy": float((y_pred_np[mask] == y_true_np[mask]).mean()),
                "recall_macro": float(
                    recall_score(y_true_np[mask], y_pred_np[mask], average="macro", zero_division=0)
                ),
            }
        )
    if int(rare_mask.sum()) > 0:
        slices.append(
            {
                "slice": f"rare(<{rare_threshold})",
                "support": int(rare_mask.sum()),
                "accuracy": float((y_pred_np[rare_mask] == y_true_np[rare_mask]).mean()),
                "recall_macro": float(
                    recall_score(
                        y_true_np[rare_mask], y_pred_np[rare_mask], average="macro", zero_division=0
                    )
                ),
            }
        )
    return slices


def apply_logit_adjustment(
    logits: torch.Tensor,
    train_class_counts: torch.Tensor | np.ndarray,
    tau: float,
) -> torch.Tensor:
    if tau == 0.0:
        return logits
    counts = torch.as_tensor(train_class_counts, dtype=torch.float32, device=logits.device)
    priors = (counts + 1.0) / (counts.sum() + len(counts))
    log_priors = torch.log(priors)
    return logits - float(tau) * log_priors.unsqueeze(0)


def fit_logit_adjustment_tau(
    val_logits: torch.Tensor,
    y_val: np.ndarray,
    train_class_counts: torch.Tensor | np.ndarray,
    tau_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> tuple[float, float]:
    y_val_np = np.asarray(y_val, dtype=np.int64)
    best_tau = 0.0
    best_f1 = -1.0
    for tau in tau_grid:
        adjusted = apply_logit_adjustment(val_logits, train_class_counts, tau)
        pred = adjusted.argmax(dim=1).cpu().numpy()
        score = float(f1_score(y_val_np, pred, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = tau
    return best_tau, best_f1


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 300,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    point = float(metric_fn(y_true_np, y_pred_np))
    rng = np.random.default_rng(seed)
    n = len(y_true_np)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(metric_fn(y_true_np[idx], y_pred_np[idx]))
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, low, high
