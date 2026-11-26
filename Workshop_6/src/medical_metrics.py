"""Clinical evaluation metrics, threshold optimization, calibration, DCA, fairness, and drift.

Implements Phase 3 core metric primitives for Workshop 6:
- Binary clinical metrics: Sensitivity, Specificity, PPV, NPV, AUC-ROC, PR-AUC, Brier, ECE, FN/FP per 1000.
- Multiclass ISIC 2018 metrics: Balanced Accuracy (macro-recall), Macro-F1, OvR ROC-AUC, PR-AUC.
- Post-hoc Temperature Scaling (Guo et al., 2017) fitted strictly on validation logits.
- Clinical threshold analysis (High-Sensitivity Screening >=95%, Youden-J Balanced, High-Specificity >=90%).
- Decision Curve Analysis & Net Benefit (Vickers & Elkin, 2006) vs Treat-All and Treat-None.
- Subgroup fairness & hidden stratification audit (Oakden-Rayner et al., 2020; Disparate Impact,
  Equal Opportunity Gap, Equalized Odds Gap).
- Production drift detection (PSI + Kolmogorov-Smirnov 2-sample test + retraining triggers).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.metrics import (  # noqa: E402
    compute_ece,
    compute_pr_auc_macro,
    compute_roc_auc_ovr_macro,
)
from Workshop_6.src.config import (  # noqa: E402
    CLASS_NAMES,
    CONFIRMATION_SPECIFICITY_TARGET,
    MALIGNANT_CLASS_IDS,
    MELANOMA_CLASS_ID,
    SCREENING_SENSITIVITY_TARGET,
    SEED,
)

# ---------------------------------------------------------------------------
# Clinical Binary & Multiclass Metrics
# ---------------------------------------------------------------------------


def binary_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary disease probabilities."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        mask = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def compute_binary_clinical_metrics(
    y_true_bin: np.ndarray,
    probs_pos: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute clinically actionable metrics at a specific decision threshold."""
    y = np.asarray(y_true_bin, dtype=np.int64)
    p = np.clip(np.asarray(probs_pos, dtype=np.float64), 0.0, 1.0)
    pred = (p >= float(threshold)).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = max(1, int(tn + fp + fn + tp))

    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    ppv = float(tp / max(tp + fp, 1))
    npv = float(tn / max(tn + fn, 1))
    bal_acc = 0.5 * (sens + spec)
    acc = float((tp + tn) / n)
    f1 = float(2 * tp / max(2 * tp + fp + fn, 1))

    has_both = len(np.unique(y)) == 2
    auc_roc = float(roc_auc_score(y, p)) if has_both else float("nan")
    pr_auc = float(average_precision_score(y, p)) if has_both else float("nan")
    brier = float(brier_score_loss(y, p)) if n > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "n_samples": float(n),
        "prevalence": float(y.mean()) if n > 0 else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": binary_ece(y, p),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "fn_per_1000": float(fn * 1000.0 / n),
        "fp_per_1000": float(fp * 1000.0 / n),
    }


def malignant_probabilities(probs_multiclass: np.ndarray) -> np.ndarray:
    """Aggregate probabilities over malignant/premalignant categories (akiec + bcc + mel)."""
    p = np.asarray(probs_multiclass, dtype=np.float64)
    return np.clip(p[:, list(MALIGNANT_CLASS_IDS)].sum(axis=1), 0.0, 1.0)


def compute_multiclass_clinical_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute ISIC 2018 7-class metrics + clinical binary tasks (malignancy & melanoma)."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)

    n_cls = len(class_names)
    per_class: dict[str, dict[str, float]] = {}
    spec_list: list[float] = []
    sens_list: list[float] = []

    for c_idx, cname in enumerate(class_names):
        y_bin = (y == c_idx).astype(np.int64)
        pred_bin = (pred == c_idx).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        ppv = float(tp / max(tp + fp, 1))
        npv = float(tn / max(tn + fn, 1))
        has_both = len(np.unique(y_bin)) == 2
        per_class[cname] = {
            "support": float(y_bin.sum()),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "auc_roc": float(roc_auc_score(y_bin, p[:, c_idx])) if has_both else float("nan"),
            "pr_auc": float(average_precision_score(y_bin, p[:, c_idx]))
            if has_both
            else float("nan"),
        }
        if y_bin.sum() > 0:
            sens_list.append(sens)
            spec_list.append(spec)

    # Multiclass Brier score (mean squared error across probability simplex)
    onehot = np.eye(n_cls, dtype=np.float64)[y]
    brier_multi = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

    # Binary clinical aggregations
    y_mal = np.isin(y, list(MALIGNANT_CLASS_IDS)).astype(np.int64)
    p_mal = malignant_probabilities(p)
    y_mel = (y == MELANOMA_CLASS_ID).astype(np.int64)
    p_mel = p[:, MELANOMA_CLASS_ID]

    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_sensitivity": float(np.mean(sens_list)) if sens_list else float("nan"),
        "macro_specificity": float(np.mean(spec_list)) if spec_list else float("nan"),
        "auc_roc_ovr_macro": compute_roc_auc_ovr_macro(y, p),
        "pr_auc_macro": compute_pr_auc_macro(y, p),
        "ece": compute_ece(y, p),
        "brier_multiclass": brier_multi,
        "malignant_default": compute_binary_clinical_metrics(y_mal, p_mal, threshold=0.5),
        "melanoma_default": compute_binary_clinical_metrics(y_mel, p_mel, threshold=0.5),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Post-hoc Calibration: Temperature Scaling (fitted strictly on Validation)
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Single-parameter Temperature Scaling for probability calibration (Guo et al., ICML 2017)."""

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor([float(np.log(max(init_temp, 1e-3)))], dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).clamp(0.05, 10.0).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temperature).clamp(0.05, 10.0)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, y_val: torch.Tensor, max_iter: int = 50) -> float:
        """Fit scalar temperature on validation logits by minimizing cross-entropy (NLL)."""
        # .clone() materializes normal autograd tensors even if caller used torch.inference_mode()
        logits_f = val_logits.detach().cpu().float().clone()
        targets = y_val.detach().cpu().long().clone()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_f), targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.temperature


# ---------------------------------------------------------------------------
# Clinical Threshold Optimization (Val-only Selection -> Test Evaluation)
# ---------------------------------------------------------------------------


def select_clinical_thresholds_on_val(
    y_val_bin: np.ndarray,
    val_probs: np.ndarray,
    *,
    screening_sens_target: float = SCREENING_SENSITIVITY_TARGET,
    confirmation_spec_target: float = CONFIRMATION_SPECIFICITY_TARGET,
) -> dict[str, float]:
    """Select 3 clinically meaningful operating thresholds strictly on validation data."""
    y = np.asarray(y_val_bin, dtype=np.int64)
    p = np.clip(np.asarray(val_probs, dtype=np.float64), 0.0, 1.0)
    if len(np.unique(y)) < 2:
        return {
            "screening_high_sens": 0.2,
            "balanced_youden": 0.5,
            "confirmation_high_spec": 0.7,
        }

    fpr, tpr, thresholds = roc_curve(y, p)
    spec = 1.0 - fpr
    finite = np.isfinite(thresholds)
    fpr, tpr, spec, thresholds = fpr[finite], tpr[finite], spec[finite], thresholds[finite]

    if len(thresholds) == 0:
        return {
            "screening_high_sens": 0.2,
            "balanced_youden": 0.5,
            "confirmation_high_spec": 0.7,
        }

    # 1. High-sensitivity screening: sensitivity >= target with maximum specificity
    ok_sens = np.where(tpr >= screening_sens_target)[0]
    if len(ok_sens) > 0:
        idx_scr = ok_sens[int(np.argmax(spec[ok_sens]))]
        thr_screening = float(thresholds[idx_scr])
    else:
        thr_screening = float(thresholds[int(np.argmax(tpr))])

    # 2. Balanced operating point: maximize Youden's J = sensitivity + specificity - 1
    youden = tpr + spec - 1.0
    thr_balanced = float(thresholds[int(np.argmax(youden))])

    # 3. High-specificity confirmation: specificity >= target with maximum sensitivity
    ok_spec = np.where(spec >= confirmation_spec_target)[0]
    if len(ok_spec) > 0:
        idx_conf = ok_spec[int(np.argmax(tpr[ok_spec]))]
        thr_confirmation = float(thresholds[idx_conf])
    else:
        thr_confirmation = float(thresholds[int(np.argmax(spec))])

    return {
        "screening_high_sens": float(np.clip(thr_screening, 0.001, 0.999)),
        "balanced_youden": float(np.clip(thr_balanced, 0.001, 0.999)),
        "confirmation_high_spec": float(np.clip(thr_confirmation, 0.001, 0.999)),
    }


def evaluate_threshold_operating_points(
    y_true_bin: np.ndarray,
    probs_pos: np.ndarray,
    thresholds_dict: dict[str, float],
    *,
    n_boot: int = 500,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Evaluate fixed validation-chosen thresholds on test set with stratified 95% bootstrap CIs."""
    rows: list[dict[str, Any]] = []
    for point_name, thr in thresholds_dict.items():
        m = compute_binary_clinical_metrics(y_true_bin, probs_pos, threshold=thr)
        cis = bootstrap_clinical_ci(y_true_bin, probs_pos, threshold=thr, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "operating_point": point_name,
                **m,
                "sensitivity_ci_low": cis["sensitivity"][1],
                "sensitivity_ci_high": cis["sensitivity"][2],
                "specificity_ci_low": cis["specificity"][1],
                "specificity_ci_high": cis["specificity"][2],
                "auc_roc_ci_low": cis["auc_roc"][1],
                "auc_roc_ci_high": cis["auc_roc"][2],
                "pr_auc_ci_low": cis["pr_auc"][1],
                "pr_auc_ci_high": cis["pr_auc"][2],
            }
        )
    return rows


def bootstrap_clinical_ci(
    y_true_bin: np.ndarray,
    probs_pos: np.ndarray,
    *,
    threshold: float = 0.5,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    """Stratified bootstrap 95% CIs for sensitivity, specificity, AUC-ROC, and PR-AUC."""
    y = np.asarray(y_true_bin, dtype=np.int64)
    p = np.asarray(probs_pos, dtype=np.float64)
    point = compute_binary_clinical_metrics(y, p, threshold=threshold)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0 or n_boot <= 0:
        return {
            k: (point[k], float("nan"), float("nan"))
            for k in ("sensitivity", "specificity", "auc_roc", "pr_auc")
        }

    rng = np.random.default_rng(seed)
    keys = ("sensitivity", "specificity", "auc_roc", "pr_auc")
    samples: dict[str, np.ndarray] = {k: np.empty(n_boot, dtype=np.float64) for k in keys}

    for i in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        bm = compute_binary_clinical_metrics(y[idx], p[idx], threshold=threshold)
        for k in keys:
            samples[k][i] = bm[k]

    out: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        vals = samples[k][np.isfinite(samples[k])]
        lo = float(np.quantile(vals, alpha / 2.0)) if len(vals) else float("nan")
        hi = float(np.quantile(vals, 1.0 - alpha / 2.0)) if len(vals) else float("nan")
        out[k] = (point[k], lo, hi)
    return out


# ---------------------------------------------------------------------------
# Decision Curve Analysis & Net Benefit (Vickers & Elkin, 2006)
# ---------------------------------------------------------------------------


def compute_decision_curve(
    y_true_bin: np.ndarray,
    probs_pos: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute Net Benefit across clinical threshold probabilities vs Treat-All & Treat-None."""
    y = np.asarray(y_true_bin, dtype=np.int64)
    p = np.clip(np.asarray(probs_pos, dtype=np.float64), 0.0, 1.0)
    n = max(1, len(y))
    prevalence = float(y.mean())

    if thresholds is None:
        thresholds = np.linspace(0.02, 0.50, 49)

    rows = []
    for pt in thresholds:
        pt_f = float(pt)
        odds = pt_f / (1.0 - pt_f)
        pred = (p >= pt_f).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())

        nb_model = (tp / n) - (fp / n) * odds
        nb_all = prevalence - (1.0 - prevalence) * odds
        nb_none = 0.0
        # Net reduction in unnecessary biopsies per 100 patients vs Treat-All
        interventions_avoided_per_100 = ((nb_model - nb_all) / max(odds, 1e-12)) * 100.0
        rows.append(
            {
                "threshold_prob": pt_f,
                "net_benefit_model": float(nb_model),
                "net_benefit_treat_all": float(nb_all),
                "net_benefit_treat_none": float(nb_none),
                "interventions_avoided_per_100": float(interventions_avoided_per_100),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subgroup Fairness & Hidden Stratification Audit (Oakden-Rayner et al., 2020)
# ---------------------------------------------------------------------------


def audit_subgroup_fairness(
    df_meta: pd.DataFrame,
    y_true_bin: np.ndarray,
    probs_pos: np.ndarray,
    *,
    threshold: float,
    subgroup_cols: tuple[str, ...] = ("age_group", "sex", "localization_group", "dx_type"),
    min_support: int = 20,
    disparity_flag_delta: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Compute disaggregated clinical metrics and group fairness gaps across metadata slices."""
    y = np.asarray(y_true_bin, dtype=np.int64)
    p = np.asarray(probs_pos, dtype=np.float64)
    overall = compute_binary_clinical_metrics(y, p, threshold=threshold)

    rows: list[dict[str, Any]] = []
    summary_gaps: dict[str, dict[str, float]] = {}

    for col in subgroup_cols:
        if col not in df_meta.columns:
            continue
        col_vals = df_meta[col].fillna("unknown").astype(str).to_numpy()
        pos_rates: list[float] = []
        tprs: list[float] = []
        fprs: list[float] = []

        for val in sorted(np.unique(col_vals)):
            mask = col_vals == val
            n_sub = int(mask.sum())
            if n_sub < min_support:
                continue
            sub_m = compute_binary_clinical_metrics(y[mask], p[mask], threshold=threshold)
            pred_sub = (p[mask] >= threshold).astype(np.int64)
            pos_rate = float(pred_sub.mean())
            tpr = sub_m["sensitivity"]
            fpr = 1.0 - sub_m["specificity"]
            n_pos = int(y[mask].sum())

            if n_pos >= 5:
                tprs.append(tpr)
            if (n_sub - n_pos) >= 5:
                fprs.append(fpr)
            pos_rates.append(pos_rate)

            sens_gap = tpr - overall["sensitivity"]
            auc_gap = (
                sub_m["auc_roc"] - overall["auc_roc"]
                if np.isfinite(sub_m["auc_roc"])
                else float("nan")
            )
            hidden_flag = bool(
                (n_pos >= 5 and sens_gap < -disparity_flag_delta)
                or (np.isfinite(auc_gap) and auc_gap < -disparity_flag_delta)
            )
            rows.append(
                {
                    "attribute": col,
                    "subgroup": str(val),
                    "n_samples": n_sub,
                    "n_positive": n_pos,
                    "prevalence": sub_m["prevalence"],
                    "positive_pred_rate": pos_rate,
                    "sensitivity": tpr,
                    "specificity": sub_m["specificity"],
                    "ppv": sub_m["ppv"],
                    "npv": sub_m["npv"],
                    "auc_roc": sub_m["auc_roc"],
                    "pr_auc": sub_m["pr_auc"],
                    "fnr": 1.0 - tpr,
                    "sens_vs_overall": sens_gap,
                    "auc_vs_overall": auc_gap,
                    "hidden_stratification_alert": hidden_flag,
                }
            )

        di = (
            float(min(pos_rates) / max(pos_rates))
            if pos_rates and max(pos_rates) > 0
            else float("nan")
        )
        eo_gap = float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
        fpr_gap = float(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
        summary_gaps[col] = {
            "disparate_impact_ratio": di,
            "equal_opportunity_gap": eo_gap,
            "fpr_gap": fpr_gap,
            "equalized_odds_gap": max(eo_gap, fpr_gap),
        }

    return pd.DataFrame(rows), summary_gaps


# ---------------------------------------------------------------------------
# Clinical Drift Monitoring (PSI + KS test + Retraining Triggers)
# ---------------------------------------------------------------------------


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index (<0.1 stable, 0.1-0.2 moderate, >0.2 retrain)."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    breakpoints = np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        breakpoints = np.linspace(float(ref.min()), float(ref.max()) + 1e-6, n_bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(ref, bins=breakpoints)[0]
    cur_counts = np.histogram(cur, bins=breakpoints)[0]
    k = len(ref_counts)
    ref_pct = (ref_counts + 1.0) / (len(ref) + k)
    cur_pct = (cur_counts + 1.0) / (len(cur) + k)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift_test(
    reference: np.ndarray, current: np.ndarray, *, significance: float = 0.05
) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test for distribution shift."""
    stat, p_value = stats.ks_2samp(
        np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    )
    return {
        "ks_statistic": float(stat),
        "ks_p_value": float(p_value),
        "ks_drift_detected": bool(p_value < significance),
    }


def evaluate_production_drift_signal(
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    *,
    scenario: str,
    psi_warn: float = 0.10,
    psi_retrain: float = 0.20,
) -> dict[str, Any]:
    """Combine PSI and KS test into a clinical monitoring status & retraining trigger."""
    psi = calculate_psi(ref_scores, cur_scores)
    ks = ks_drift_test(ref_scores, cur_scores)
    if psi >= psi_retrain:
        action = "RETRAIN_ALERT"
    elif psi >= psi_warn or ks["ks_drift_detected"]:
        action = "INVESTIGATE"
    else:
        action = "OK"
    return {
        "scenario": scenario,
        "psi": psi,
        **ks,
        "recommended_action": action,
    }


__all__ = [
    "TemperatureScaler",
    "audit_subgroup_fairness",
    "binary_ece",
    "bootstrap_clinical_ci",
    "calculate_psi",
    "compute_binary_clinical_metrics",
    "compute_decision_curve",
    "compute_multiclass_clinical_metrics",
    "evaluate_production_drift_signal",
    "evaluate_threshold_operating_points",
    "ks_drift_test",
    "malignant_probabilities",
    "select_clinical_thresholds_on_val",
]
