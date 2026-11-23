"""Pruebas unitarias de shared/metrics.py."""

from __future__ import annotations

import numpy as np
import torch

from shared.metrics import (
    apply_logit_adjustment,
    bootstrap_ci,
    compute_multiclass_metrics,
    fit_logit_adjustment_tau,
    slice_metrics_by_char_type,
)


def test_multiclass_metrics_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 3, 1, 2, 0, 3])
    logits = torch.full((8, 4), -10.0)
    for i, cls in enumerate(y):
        logits[i, cls] = 10.0

    m = compute_multiclass_metrics(y, logits, with_ranking_metrics=True)
    assert m["accuracy"] == 1.0
    assert m["top5_accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["pr_auc_macro"] == 1.0
    assert m["bpc"] < 1e-4
    assert m["ece"] < 1e-3


def test_slice_metrics_and_bootstrap_ci(synthetic_vocab) -> None:
    itos = synthetic_vocab["itos"]
    y_true = np.array([1, 1, 2, 2, 8, 9])
    y_pred = np.array([1, 1, 2, 1, 8, 2])
    slices = slice_metrics_by_char_type(y_true, y_pred, itos, rare_threshold=2)
    slice_names = {s["slice"] for s in slices}
    assert {"space", "lowercase", "uppercase", "punctuation/other"}.issubset(slice_names)

    point, low, high = bootstrap_ci(y_true, y_pred, lambda yt, yp: float((yt == yp).mean()), n_boot=50)
    assert 0.0 <= low <= point <= high <= 1.0


def test_logit_adjustment_boosts_rare_class_f1() -> None:
    counts = np.array([950, 50])
    y_val = np.array([0] * 95 + [1] * 5)
    logits = torch.zeros((100, 2))
    logits[:95, 0] = 3.0
    logits[95:, 0] = 0.6
    logits[95:, 1] = 0.2

    tau, best_f1 = fit_logit_adjustment_tau(logits, y_val, counts)
    base_metrics = compute_multiclass_metrics(y_val, logits, with_ranking_metrics=False)
    adj_logits = apply_logit_adjustment(logits, counts, tau)
    adj_metrics = compute_multiclass_metrics(y_val, adj_logits, with_ranking_metrics=False)

    assert tau > 0.0
    assert best_f1 >= base_metrics["f1_macro"]
    assert adj_metrics["f1_macro"] > base_metrics["f1_macro"]
