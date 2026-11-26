"""Unit tests for clinical metrics, temperature scaling, thresholds, DCA, fairness, and drift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_6.src.medical_metrics import (  # noqa: E402
    TemperatureScaler,
    audit_subgroup_fairness,
    calculate_psi,
    compute_binary_clinical_metrics,
    compute_decision_curve,
    compute_multiclass_clinical_metrics,
    evaluate_production_drift_signal,
    select_clinical_thresholds_on_val,
)


def test_binary_clinical_metrics_exact_confusion_values() -> None:
    # TP=3, FN=1, FP=1, TN=5 -> Sens=0.75, Spec=5/6, PPV=0.75, NPV=5/6
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    m = compute_binary_clinical_metrics(y_true, probs, threshold=0.5)
    assert np.isclose(m["sensitivity"], 0.75)
    assert np.isclose(m["specificity"], 5.0 / 6.0)
    assert np.isclose(m["ppv"], 0.75)
    assert np.isclose(m["npv"], 5.0 / 6.0)
    assert m["fn_per_1000"] == 100.0
    assert np.isclose(m["auc_roc"], 20.5 / 24.0)


def test_temperature_scaling_improves_overconfident_nll() -> None:
    torch.manual_seed(42)
    y = torch.randint(0, 7, (200,))
    # Artificially overconfident logits (scaled by 5.0)
    base_logits = torch.randn(200, 7)
    base_logits[torch.arange(200), y] += 1.2
    overconfident = base_logits * 5.0

    ce = torch.nn.CrossEntropyLoss()
    nll_before = float(ce(overconfident, y).item())
    scaler = TemperatureScaler()
    t = scaler.fit(overconfident, y)
    nll_after = float(ce(overconfident / t, y).item())

    assert t > 1.5
    assert nll_after < nll_before


def test_clinical_thresholds_respect_operating_order_and_dca_net_benefit() -> None:
    rng = np.random.default_rng(42)
    y = np.concatenate([np.zeros(400, dtype=np.int64), np.ones(100, dtype=np.int64)])
    p = np.concatenate([rng.beta(2, 5, size=400), rng.beta(5, 2, size=100)])

    thrs = select_clinical_thresholds_on_val(y, p)
    m_scr = compute_binary_clinical_metrics(y, p, threshold=thrs["screening_high_sens"])
    m_conf = compute_binary_clinical_metrics(y, p, threshold=thrs["confirmation_high_spec"])
    assert thrs["screening_high_sens"] <= thrs["balanced_youden"] <= thrs["confirmation_high_spec"]
    assert m_scr["sensitivity"] >= 0.95
    assert m_conf["specificity"] >= 0.90

    dca = compute_decision_curve(y, p)
    mid = dca[(dca["threshold_prob"] >= 0.15) & (dca["threshold_prob"] <= 0.35)]
    assert (mid["net_benefit_model"] > mid["net_benefit_treat_all"]).all()
    assert (mid["net_benefit_model"] > 0.0).all()


def test_subgroup_fairness_flags_hidden_stratification() -> None:
    # Group A has high sensitivity (0.95); Group B fails systematically (0.20)
    n = 100
    df = pd.DataFrame(
        {
            "age_group": ["40-59"] * n + [">=75"] * n,
            "sex": ["female"] * (2 * n),
        }
    )
    y = np.concatenate([np.array([1] * 20 + [0] * 80), np.array([1] * 20 + [0] * 80)])
    p_a = np.array([0.85] * 19 + [0.1] + [0.1] * 80)
    p_b = np.array([0.85] * 4 + [0.1] * 16 + [0.1] * 80)
    p = np.concatenate([p_a, p_b])

    fair_df, gaps = audit_subgroup_fairness(
        df, y, p, threshold=0.5, subgroup_cols=("age_group",), min_support=20
    )
    elderly = fair_df[fair_df["subgroup"] == ">=75"].iloc[0]
    assert bool(elderly["hidden_stratification_alert"]) is True
    assert gaps["age_group"]["equal_opportunity_gap"] >= 0.70


def test_drift_psi_and_ks_distinguish_stable_vs_shifted() -> None:
    rng = np.random.default_rng(42)
    ref = rng.beta(2, 6, size=1000)
    cur_ok = rng.beta(2, 6, size=500)
    cur_shift = rng.beta(5, 3, size=500)

    sig_ok = evaluate_production_drift_signal(ref, cur_ok, scenario="ok")
    sig_shift = evaluate_production_drift_signal(ref, cur_shift, scenario="shift")

    assert calculate_psi(ref, cur_ok) < 0.10
    assert sig_ok["recommended_action"] == "OK"
    assert sig_shift["psi"] > 0.20
    assert sig_shift["ks_drift_detected"] is True
    assert sig_shift["recommended_action"] == "RETRAIN_ALERT"


def test_multiclass_clinical_metrics_returns_all_7_classes() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 7, size=140)
    probs = rng.dirichlet(np.ones(7), size=140)
    out = compute_multiclass_clinical_metrics(y, probs)
    assert len(out["per_class"]) == 7
    assert "balanced_accuracy" in out and "malignant_default" in out
