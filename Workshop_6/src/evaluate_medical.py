"""Comprehensive checkpoint-driven clinical evaluation, fairness audit, drift simulation, and master W1-W6 table.

Implements Phase 3 of Workshop 6:
- Reproduces 100% of reported medical test metrics strictly from saved checkpoints (`weights_only=True`).
- Evaluates all ablation models (A1-A4) + W4 `MedConvVAE2D` anomaly detector on the untouched test split.
- Computes clinical metrics (Sensitivity, Specificity, PPV, NPV, AUC-ROC, PR-AUC, Balanced Accuracy,
  ECE, Brier, DCA/Net Benefit) at 3 validation-chosen clinical operating points with 95% bootstrap CIs.
- Runs subgroup fairness & hidden stratification audit (`age_group`, `sex`, `localization_group`, `dx_type`).
- Runs simulated clinical deployment drift tests (PSI + KS test) under scanner illumination, blur, and shift.
- Builds the Master Comparison Table (W1-W6 + published HAM10000/ISIC/DermaMNIST benchmarks + dermatologists).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint  # noqa: E402
from shared.device import (  # noqa: E402
    DEVICE_CHOICES,
    dataloader_kwargs,
    get_amp_config,
    resolve_device,
    seed_everything,
)
from shared.metrics import apply_logit_adjustment  # noqa: E402
from Workshop_6.src.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    FIGURES_DIR,
    REPORTS_DIR,
    SEED,
    VAE_IMAGE_SIZE,
)
from Workshop_6.src.medical_metrics import (  # noqa: E402
    audit_subgroup_fairness,
    compute_binary_clinical_metrics,
    compute_decision_curve,
    compute_multiclass_clinical_metrics,
    evaluate_production_drift_signal,
    evaluate_threshold_operating_points,
    malignant_probabilities,
)
from Workshop_6.src.medical_preprocessing import (  # noqa: E402
    HAM10000Dataset,
    build_medical_transforms,
    get_cache_dir,
    load_split_dataframes,
)
from Workshop_6.src.transfer import (  # noqa: E402
    MedConvVAE2D,
    build_medical_classifier,
    predict_classifier_logits,
    score_vae_anomalies,
)

# Verified published medical imaging benchmarks & human reader performance (with DOIs in references.bib)
PUBLISHED_BENCHMARKS: list[dict[str, Any]] = [
    {
        "stage": "Literature (Human)",
        "domain": "HAM10000 / ISIC 2018",
        "model": "511 Human Readers (283 Board-Certified Dermatologists, Tschandl et al. 2019 Lancet Oncol.)",
        "transfer_status": "Clinical Gold Reference",
        "primary_metric": "Mean correct dx: 17.91/30 (59.7%); Expert >10y: 18.78/30 (62.6%); Malignancy Sens ~89%, Spec ~75%",
        "balanced_accuracy": 0.626,
        "malignant_auc_roc": float("nan"),
        "notes": "DOI: 10.1016/S1470-2045(19)30333-X (30-image randomized ISIC 2018 batches)",
    },
    {
        "stage": "Literature (ML)",
        "domain": "HAM10000 / ISIC 2018",
        "model": "ISIC 2018 Task 3 Challenge Top-1 Ensemble (Codella et al. 2019 / Tschandl et al. 2019)",
        "transfer_status": "Published Benchmark",
        "primary_metric": "Balanced Multi-class Acc: 0.885 (leaderboard top-1); Mean top-3 AI: 25.43/30 correct",
        "balanced_accuracy": 0.885,
        "malignant_auc_roc": 0.960,
        "notes": "arXiv:1902.03368 / Lancet Oncology 2019 (heavy multi-architecture ensembles + external data)",
    },
    {
        "stage": "Literature (ML)",
        "domain": "DermaMNIST-224 (HAM10000)",
        "model": "ResNet-18 (224x224) MedMNIST v2 Baseline (Yang et al. 2023 Scientific Data)",
        "transfer_status": "Published Benchmark",
        "primary_metric": "Test Acc: 0.754 | OvR ROC-AUC: 0.920",
        "balanced_accuracy": float("nan"),
        "malignant_auc_roc": 0.920,
        "notes": "DOI: 10.1038/s41597-022-01721-8 (single ResNet-18 without dermoscopy-specific preproc)",
    },
]


def _load_calibrated_test_predictions(
    ckpt_path: Path,
    test_df: pd.DataFrame,
    split_info: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Load checkpoint (`weights_only=True`) and compute raw + calibrated test probabilities."""
    sd, meta = load_checkpoint(ckpt_path, device="cpu")
    model = build_medical_classifier(meta["model_hparams"]).to(device).eval()
    model.load_state_dict(sd)

    img_sz = int(meta["model_hparams"].get("image_size", DEFAULT_IMAGE_SIZE))
    enable_pre = bool(meta.get("enable_medical_preproc", True))
    st = split_info["train_stats"]["medical_preproc" if enable_pre else "raw"]
    tf = build_medical_transforms(img_sz, is_train=False, mean=st["mean"], std=st["std"])
    cache = get_cache_dir(img_sz, enable_medical_preproc=enable_pre)

    ds = HAM10000Dataset(
        test_df,
        image_size=img_sz,
        enable_medical_preproc=enable_pre,
        transform=tf,
        cache_dir=cache if cache.exists() else None,
    )
    loader = DataLoader(
        ds, batch_size=64, shuffle=False, **dataloader_kwargs(device, num_workers=0)
    )
    amp_en, amp_dt, _ = get_amp_config(device)
    logits, y_true = predict_classifier_logits(
        model, loader, device, amp_enabled=amp_en, amp_dtype=amp_dt
    )

    cal = meta.get("calibration", {})
    temp = float(cal.get("temperature", 1.0))
    tau = float(cal.get("logit_adj_tau", 0.0))
    counts = np.asarray(cal.get("train_class_counts", [1] * len(CLASS_NAMES)))

    probs_raw = torch.softmax(logits, dim=-1).numpy()
    adj_logits = apply_logit_adjustment(logits, counts, tau) if tau > 0.0 else logits
    probs_cal = torch.softmax(adj_logits / temp, dim=-1).numpy()
    return meta, y_true, probs_raw, probs_cal


def _read_original_domain_w1_w5_rows() -> list[dict[str, Any]]:
    """Read real test metrics from Workshop_1..Workshop_5 CSVs already in the repository."""
    rows: list[dict[str, Any]] = []

    # Workshop 1
    w1_csv = REPO_ROOT / "Workshop_1" / "reports" / "baselines_summary.csv"
    if w1_csv.exists():
        df1 = pd.read_csv(w1_csv)
        for _, r in df1.iterrows():
            rows.append(
                {
                    "stage": "W1 (Original)",
                    "domain": "Shakespeare (76-char)",
                    "model": f"W1 {r['model']}",
                    "transfer_status": "Methodology only (GroupSplit)",
                    "primary_metric": (
                        f"Acc={r['test_accuracy']:.4f}, F1-mac={r['test_f1_macro']:.4f}, "
                        f"AUC={r['test_auc_roc_ovr_macro']:.4f}, BPC={r['test_bpc']:.3f}"
                    ),
                    "balanced_accuracy": float("nan"),
                    "malignant_auc_roc": float("nan"),
                    "notes": "Char n-gram / morphological features do NOT transfer to 2D images",
                }
            )

    # Workshop 2
    w2_csv = REPO_ROOT / "Workshop_2" / "reports" / "results_comparison.csv"
    if w2_csv.exists():
        df2 = pd.read_csv(w2_csv)
        for _, r in df2[df2["split"] == "test_full"].iterrows():
            rows.append(
                {
                    "stage": "W2 (Original)",
                    "domain": "Shakespeare (76-char)",
                    "model": str(r["model"]),
                    "transfer_status": "Hybrid Arch + Logit Adj transferred",
                    "primary_metric": (
                        f"Acc={r['accuracy']:.4f}, Top5={r['top5_accuracy']:.4f}, "
                        f"F1-mac={r['f1_macro']:.4f}, BPC={r['bpc']:.4f}, ECE={r['ece']:.4f}"
                    ),
                    "balanced_accuracy": float("nan"),
                    "malignant_auc_roc": float("nan"),
                    "notes": "Pre-LN Transformer blocks & Logit Adj adapted; char embeddings discarded",
                }
            )

    # Workshop 3
    w3_csv = REPO_ROOT / "Workshop_3" / "reports" / "test_comparison.csv"
    if w3_csv.exists():
        df3 = pd.read_csv(w3_csv)
        dqn = df3[df3["policy"].str.contains("DQN", case=False, na=False)]
        if len(dqn) > 0:
            r = dqn.iloc[0]
            rows.append(
                {
                    "stage": "W3 (Original)",
                    "domain": "Shakespeare (RL Env)",
                    "model": str(r["policy"]),
                    "transfer_status": "Does NOT transfer (Honest Negative)",
                    "primary_metric": (
                        f"Return={r['mean_return']:+.3f}, NewWord={r['new_word_rate']:.3f}, "
                        f"Invalid={r['invalid_word_rate']:.4f}, Quality={r['composite_quality']:.3f}"
                    ),
                    "balanced_accuracy": float("nan"),
                    "malignant_auc_roc": float("nan"),
                    "notes": "Single-step medical image diagnosis is not a lexical MDP",
                }
            )

    # Workshop 4
    w4_rec = REPO_ROOT / "Workshop_4" / "reports" / "reconstruction_metrics.csv"
    w4_anom = REPO_ROOT / "Workshop_4" / "reports" / "anomaly_detection_metrics.csv"
    if w4_rec.exists() and w4_anom.exists():
        r4 = pd.read_csv(w4_rec)
        a4 = pd.read_csv(w4_anom)
        vae_r = r4[r4["model"].str.contains("VAE", na=False)].iloc[0]
        vae_a = a4[a4["method"].str.contains("recon \\+ KL", na=False)].iloc[0]
        rows.append(
            {
                "stage": "W4 (Original)",
                "domain": "Shakespeare (1D Seq)",
                "model": f"{vae_r['model']} + Anomaly",
                "transfer_status": "Adapted 1D->2D (MedConvVAE2D)",
                "primary_metric": (
                    f"ReconAcc={vae_r['char_accuracy']:.4f}, CE={vae_r['ce_per_char']:.3f}, "
                    f"Anom PR-AUC={vae_a['pr_auc']:.4f}, F1={vae_a['f1']:.4f}"
                ),
                "balanced_accuracy": float("nan"),
                "malignant_auc_roc": float("nan"),
                "notes": "GroupNorm+GELU, FiLM, Free Bits & Cyclical KL migrated to 2D continuous loss",
            }
        )

    # Workshop 5
    w5_aug = REPO_ROOT / "Workshop_5" / "reports" / "metrics" / "augmentation_results.csv"
    if w5_aug.exists():
        df5 = pd.read_csv(w5_aug)
        for _, r in df5.iterrows():
            rows.append(
                {
                    "stage": "W5 (Original)",
                    "domain": "Shakespeare (GAN Aug)",
                    "model": f"W5 Transformer ({r['condition']})",
                    "transfer_status": "Adapted 1D->2D (cWGAN-GP, A4)",
                    "primary_metric": (
                        f"BPC={r['test_bpc']:.4f}, Acc={r['test_accuracy']:.4f}, "
                        f"F1-cal={r['test_f1_macro_cal']:.4f}, F1-adj={r['test_f1_macro_adj']:.4f}"
                    ),
                    "balanced_accuracy": float("nan"),
                    "malignant_auc_roc": float("nan"),
                    "notes": "FP32 GP & filtered aug transferred; Gumbel-ST & n-gram TV discarded",
                }
            )
    return rows


def _simulate_clinical_drift_scenarios(
    model: torch.nn.Module,
    meta: dict[str, Any],
    test_df: pd.DataFrame,
    split_info: dict[str, Any],
    ref_p_mal: np.ndarray,
    device: torch.device,
    seed: int = SEED,
) -> pd.DataFrame:
    """Simulate 4 hospital deployment shifts and run PSI + KS drift monitoring tests."""
    seed_everything(seed, deterministic=True)
    img_sz = int(meta["model_hparams"].get("image_size", DEFAULT_IMAGE_SIZE))
    enable_pre = bool(meta.get("enable_medical_preproc", True))
    st = split_info["train_stats"]["medical_preproc" if enable_pre else "raw"]
    cal = meta.get("calibration", {})
    temp = float(cal.get("temperature", 1.0))
    tau = float(cal.get("logit_adj_tau", 0.0))
    counts = np.asarray(cal.get("train_class_counts", [1] * len(CLASS_NAMES)))
    cache = get_cache_dir(img_sz, enable_medical_preproc=enable_pre)

    # Sample up to 400 test images deterministically for fast shift simulation
    n_sub = min(400, len(test_df))
    sub_df = test_df.sample(n=n_sub, random_state=seed).reset_index(drop=True)
    base_tf = build_medical_transforms(img_sz, is_train=False, mean=st["mean"], std=st["std"])

    def _infer_with_perturbation(perturb_fn) -> tuple[np.ndarray, np.ndarray]:
        probs_out: list[np.ndarray] = []
        brightness_vals: list[float] = []
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(sub_df), 32):
                batch_tensors = []
                for _, row in sub_df.iloc[i : i + 32].iterrows():
                    p = (
                        (cache / f"{row['image_id']}.png")
                        if cache.exists()
                        else Path(row["image_path"])
                    )
                    arr = np.asarray(
                        cv2.resize(
                            cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB),
                            (img_sz, img_sz),
                        ),
                        dtype=np.uint8,
                    )
                    arr = perturb_fn(arr)
                    brightness_vals.append(float(arr.mean() / 255.0))
                    from PIL import Image

                    batch_tensors.append(base_tf(Image.fromarray(arr)))
                xb = torch.stack(batch_tensors, dim=0).to(device)
                logits = model(xb).float().cpu()
                if tau > 0.0:
                    logits = apply_logit_adjustment(logits, counts, tau)
                probs = torch.softmax(logits / temp, dim=-1).numpy()
                probs_out.append(malignant_probabilities(probs))
        return np.concatenate(probs_out, axis=0), np.asarray(brightness_vals, dtype=np.float64)

    def _no_shift(img: np.ndarray) -> np.ndarray:
        return img

    def _scanner_warm_shift(img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        out[..., 0] = np.clip(out[..., 0] * 1.22 + 12.0, 0, 255)
        out[..., 2] = np.clip(out[..., 2] * 0.80, 0, 255)
        return out.astype(np.uint8)

    def _optical_defocus(img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (9, 9), sigmaX=2.4)

    def _low_exposure_noise(img: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = img.astype(np.float32) * 0.68 + rng.normal(0, 10.0, size=img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    y_sub_mal = sub_df["is_malignant"].to_numpy()
    thr_bal = float(meta.get("clinical_thresholds_val", {}).get("balanced_youden", 0.5))

    ref_p_sub, ref_bright = _infer_with_perturbation(_no_shift)
    scenarios = [
        ("In-Distribution Control (No Shift)", _no_shift),
        ("New Dermatoscope Color/Illumination Shift", _scanner_warm_shift),
        ("Optical Defocus / Motion Blur (sigma=2.4)", _optical_defocus),
        ("Under-Exposure + Sensor Noise (-32% gain)", _low_exposure_noise),
    ]

    drift_rows: list[dict[str, Any]] = []
    for name, fn in scenarios:
        cur_p, cur_b = _infer_with_perturbation(fn)
        prob_sig = evaluate_production_drift_signal(ref_p_mal, cur_p, scenario=name)
        img_sig = evaluate_production_drift_signal(ref_bright, cur_b, scenario=name)
        m = compute_binary_clinical_metrics(y_sub_mal, cur_p, threshold=thr_bal)
        drift_rows.append(
            {
                "scenario": name,
                "pred_psi": prob_sig["psi"],
                "pred_ks_stat": prob_sig["ks_statistic"],
                "pred_ks_p": prob_sig["ks_p_value"],
                "image_brightness_psi": img_sig["psi"],
                "malignant_auc": m["auc_roc"],
                "malignant_sensitivity": m["sensitivity"],
                "malignant_specificity": m["specificity"],
                "recommended_action": (
                    "RETRAIN_ALERT"
                    if "RETRAIN_ALERT"
                    in (prob_sig["recommended_action"], img_sig["recommended_action"])
                    else (
                        "INVESTIGATE"
                        if "INVESTIGATE"
                        in (prob_sig["recommended_action"], img_sig["recommended_action"])
                        else "OK"
                    )
                ),
            }
        )
    return pd.DataFrame(drift_rows)


# ---------------------------------------------------------------------------
# Publication-Quality Plots (300 dpi, Colorblind-Safe Okabe-Ito Palette)
# ---------------------------------------------------------------------------


def _plot_roc_pr_curves(
    y_true_mal: np.ndarray,
    model_probs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for i, (name, p_mal) in enumerate(model_probs.items()):
        color = palette[i % len(palette)]
        fpr, tpr, _ = roc_curve(y_true_mal, p_mal)
        m = compute_binary_clinical_metrics(y_true_mal, p_mal, threshold=0.5)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={m['auc_roc']:.3f})", color=color, lw=2.0)

        prec, rec, _ = precision_recall_curve(y_true_mal, p_mal)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC={m['pr_auc']:.3f})", color=color, lw=2.0)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.0)
    axes[0].set(
        title="ROC Curve — Malignancy Detection (Test Set)",
        xlabel="1 - Specificity (FPR)",
        ylabel="Sensitivity (TPR)",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)

    prev = float(y_true_mal.mean())
    axes[1].axhline(
        prev, color="k", linestyle="--", alpha=0.4, lw=1.0, label=f"Prevalence ({prev:.2f})"
    )
    axes[1].set(
        title="Precision-Recall Curve — Malignancy Detection",
        xlabel="Sensitivity (Recall)",
        ylabel="PPV (Precision)",
    )
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_and_dca(
    y_true_mal: np.ndarray,
    p_mal_raw: np.ndarray,
    p_mal_cal: np.ndarray,
    dca_df: pd.DataFrame,
    model_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Reliability diagram
    f_raw, mp_raw = calibration_curve(y_true_mal, p_mal_raw, n_bins=10, strategy="quantile")
    f_cal, mp_cal = calibration_curve(y_true_mal, p_mal_cal, n_bins=10, strategy="quantile")
    m_raw = compute_binary_clinical_metrics(y_true_mal, p_mal_raw)
    m_cal = compute_binary_clinical_metrics(y_true_mal, p_mal_cal)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    axes[0].plot(
        mp_raw, f_raw, "o-", color="#D55E00", label=f"Uncalibrated (ECE={m_raw['ece']:.3f})"
    )
    axes[0].plot(
        mp_cal, f_cal, "s-", color="#0072B2", label=f"Temp-Scaled (ECE={m_cal['ece']:.3f})"
    )
    axes[0].set(
        title=f"Calibration Curve ({model_label})",
        xlabel="Predicted Malignancy Probability",
        ylabel="Observed Frequency",
    )
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    # Decision Curve Analysis
    axes[1].plot(
        dca_df["threshold_prob"],
        dca_df["net_benefit_model"],
        color="#0072B2",
        lw=2.2,
        label=f"Model ({model_label})",
    )
    axes[1].plot(
        dca_df["threshold_prob"],
        dca_df["net_benefit_treat_all"],
        "--",
        color="#D55E00",
        lw=1.6,
        label="Biopsy All (Treat All)",
    )
    axes[1].plot(
        dca_df["threshold_prob"],
        dca_df["net_benefit_treat_none"],
        ":",
        color="#333333",
        lw=1.6,
        label="Biopsy None (Treat None)",
    )
    axes[1].set_ylim(-0.04, max(0.22, float(dca_df["net_benefit_model"].max()) * 1.15))
    axes[1].set(
        title="Decision Curve Analysis (Clinical Net Benefit)",
        xlabel="Decision Threshold Probability (p_t)",
        ylabel="Net Benefit",
    )
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fairness_subgroups(fairness_df: pd.DataFrame, out_path: Path) -> None:
    attrs = [
        a
        for a in ("age_group", "sex", "localization_group", "dx_type", "dataset")
        if a in fairness_df["attribute"].unique()
    ]
    if not attrs:
        return
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 4.2), sharey=True)
    if len(attrs) == 1:
        axes = [axes]
    for ax, attr in zip(axes, attrs, strict=True):
        sub = fairness_df[fairness_df["attribute"] == attr]
        x = np.arange(len(sub))
        w = 0.36
        ax.bar(x - w / 2, sub["sensitivity"], width=w, color="#0072B2", label="Sensitivity")
        ax.bar(x + w / 2, sub["specificity"], width=w, color="#009E73", label="Specificity")
        for idx, (_, r) in enumerate(sub.iterrows()):
            if r["hidden_stratification_alert"]:
                ax.text(
                    idx,
                    max(r["sensitivity"], r["specificity"]) + 0.03,
                    "!",
                    color="#D55E00",
                    ha="center",
                    fontweight="bold",
                    fontsize=11,
                )
        ax.set_xticks(x, sub["subgroup"].tolist(), rotation=25, ha="right", fontsize=8.5)
        ax.set_title(attr.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle(
        "Subgroup Fairness & Hidden Stratification Audit (! = >10 pp drop)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), normalize="true")
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [c.upper() for c in CLASS_NAMES]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(title=title, xlabel="Predicted Class", ylabel="True Class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=("white" if val > 0.5 else "black"),
                fontsize=8.5,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    df = summary_df.sort_values("malignant_auc_roc", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.52 * len(df))))
    y_pos = np.arange(len(df))
    h = 0.36
    ax.barh(
        y_pos - h / 2,
        df["balanced_accuracy"],
        height=h,
        color="#0072B2",
        label="7-Class Balanced Acc",
    )
    ax.barh(
        y_pos + h / 2,
        df["malignant_auc_roc"],
        height=h,
        color="#D55E00",
        label="Malignancy AUC-ROC",
    )
    ax.set_yticks(y_pos, df["run_name"].tolist(), fontsize=8.5)
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("Test Set Score")
    ax.set_title(
        "Workshop 6 Ablation & Transfer Comparison on Untouched Test Set", fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Comprehensive Evaluation Pipeline
# ---------------------------------------------------------------------------


def evaluate_all_checkpoints(
    *,
    device_str: str = "auto",
    n_boot: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=True)
    device, desc = resolve_device(device_str)
    print(f"[Eval] Device: {device} | {desc}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dfs, split_info = load_split_dataframes()
    test_df = dfs["test"].reset_index(drop=True)
    y_test_mal = test_df["is_malignant"].to_numpy()

    classifier_ckpts = sorted(
        p for p in CHECKPOINTS_DIR.glob("a*.pt") if p.with_suffix(".meta.json").exists()
    )
    if not classifier_ckpts:
        raise FileNotFoundError(
            f"No trained classifier checkpoints (`a*.pt`) found in {CHECKPOINTS_DIR}. "
            "Run `python -m Workshop_6.src.transfer --run all` first."
        )

    eval_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    roc_curves_map: dict[str, np.ndarray] = {}
    cached_outputs: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = {}

    for ckpt in classifier_ckpts:
        meta, y_true, probs_raw, probs_cal = _load_calibrated_test_predictions(
            ckpt, test_df, split_info, device
        )
        run_name = str(meta.get("run_name", ckpt.stem))
        cached_outputs[run_name] = (meta, y_true, probs_raw, probs_cal)

        mc = compute_multiclass_clinical_metrics(y_true, probs_cal)
        p_mal = malignant_probabilities(probs_cal)
        roc_curves_map[run_name] = p_mal

        thrs = meta.get(
            "clinical_thresholds_val",
            {"screening_high_sens": 0.15, "balanced_youden": 0.35, "confirmation_high_spec": 0.65},
        )
        op_rows = evaluate_threshold_operating_points(
            y_test_mal, p_mal, thrs, n_boot=n_boot, seed=seed
        )
        for opr in op_rows:
            threshold_rows.append({"run_name": run_name, **opr})

        bal_op = next((r for r in op_rows if r["operating_point"] == "balanced_youden"), op_rows[0])
        scr_op = next(
            (r for r in op_rows if r["operating_point"] == "screening_high_sens"), op_rows[0]
        )

        eval_rows.append(
            {
                "run_name": run_name,
                "arch": meta["model_hparams"].get("arch", "unknown"),
                "params": int(meta.get("trainable_params", 0)),
                "accuracy": mc["accuracy"],
                "balanced_accuracy": mc["balanced_accuracy"],
                "f1_macro": mc["f1_macro"],
                "auc_roc_ovr_macro": mc["auc_roc_ovr_macro"],
                "pr_auc_macro": mc["pr_auc_macro"],
                "ece_cal": mc["ece"],
                "brier_multiclass": mc["brier_multiclass"],
                "malignant_auc_roc": bal_op["auc_roc"],
                "malignant_pr_auc": bal_op["pr_auc"],
                "malignant_sens_youden": bal_op["sensitivity"],
                "malignant_spec_youden": bal_op["specificity"],
                "malignant_ppv_youden": bal_op["ppv"],
                "malignant_npv_youden": bal_op["npv"],
                "malignant_sens_screening": scr_op["sensitivity"],
                "malignant_spec_screening": scr_op["specificity"],
                "melanoma_auc_roc": mc["per_class"]["mel"]["auc_roc"],
                "melanoma_sensitivity": mc["per_class"]["mel"]["sensitivity"],
                "temperature": float(meta.get("calibration", {}).get("temperature", 1.0)),
                "logit_adj_tau": float(meta.get("calibration", {}).get("logit_adj_tau", 0.0)),
            }
        )

    # Evaluate W4 MedConvVAE2D anomaly detector if present
    vae_ckpt = CHECKPOINTS_DIR / "w4_med_conv_vae2d.pt"
    vae_summary: dict[str, Any] | None = None
    if vae_ckpt.exists():
        v_sd, v_meta = load_checkpoint(vae_ckpt, device="cpu")
        vae = (
            MedConvVAE2D(
                image_size=int(v_meta["model_hparams"]["image_size"]),
                z_dim=int(v_meta["model_hparams"]["z_dim"]),
            )
            .to(device)
            .eval()
        )
        vae.load_state_dict(v_sd)
        tf_v = transforms.Compose(
            [transforms.Resize((VAE_IMAGE_SIZE, VAE_IMAGE_SIZE)), transforms.ToTensor()]
        )
        cache = get_cache_dir(DEFAULT_IMAGE_SIZE, enable_medical_preproc=True)
        ds_v = HAM10000Dataset(
            test_df,
            image_size=VAE_IMAGE_SIZE,
            enable_medical_preproc=True,
            transform=tf_v,
            cache_dir=cache if cache.exists() else None,
        )
        v_loader = DataLoader(ds_v, batch_size=64, shuffle=False, num_workers=0)
        raw_sc, y_v_mal = score_vae_anomalies(vae, v_loader, device)
        b = v_meta["anomaly_norm_bounds"]
        norm_sc = np.clip((raw_sc - b["q01"]) / max(b["q99"] - b["q01"], 1e-6), 0.0, 1.0)
        thr_v = float(v_meta["clinical_thresholds_val"]["balanced_youden"])
        vae_m = compute_binary_clinical_metrics(y_v_mal, norm_sc, threshold=thr_v)
        vae_summary = {"run_name": "w4_med_conv_vae2d", **vae_m}
        roc_curves_map["W4 MedConvVAE2D (Anomaly)"] = norm_sc
        (REPORTS_DIR / "vae_anomaly_test_metrics.json").write_text(
            json.dumps(vae_summary, indent=2) + "\n", encoding="utf-8"
        )

    summary_df = pd.DataFrame(eval_rows).sort_values("malignant_auc_roc", ascending=False)
    summary_df.to_csv(REPORTS_DIR / "clinical_metrics_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    # Identify best overall model for deep clinical diagnostics (DCA, Fairness, Drift, CM)
    best_name = str(summary_df.iloc[0]["run_name"])
    best_meta, y_best, p_raw_best, p_cal_best = cached_outputs[best_name]
    p_mal_raw_best = malignant_probabilities(p_raw_best)
    p_mal_cal_best = malignant_probabilities(p_cal_best)
    best_thr_bal = float(best_meta.get("clinical_thresholds_val", {}).get("balanced_youden", 0.35))

    # Per-class detailed table for best model
    best_mc = compute_multiclass_clinical_metrics(y_best, p_cal_best)
    per_cls_df = pd.DataFrame.from_dict(best_mc["per_class"], orient="index").reset_index()
    per_cls_df.rename(columns={"index": "dx"}, inplace=True)
    per_cls_df.to_csv(REPORTS_DIR / "best_model_per_class_metrics.csv", index=False)

    # Decision Curve Analysis
    dca_df = compute_decision_curve(y_test_mal, p_mal_cal_best)
    dca_df.to_csv(REPORTS_DIR / "decision_curve_analysis.csv", index=False)

    # Fairness & Hidden Stratification Audit
    fair_df, fair_gaps = audit_subgroup_fairness(
        test_df, y_test_mal, p_mal_cal_best, threshold=best_thr_bal
    )
    fair_df.to_csv(REPORTS_DIR / "subgroup_fairness_audit.csv", index=False)
    (REPORTS_DIR / "fairness_summary_gaps.json").write_text(
        json.dumps(fair_gaps, indent=2) + "\n", encoding="utf-8"
    )

    # Clinical Drift Monitoring Simulation
    best_sd, _ = load_checkpoint(CHECKPOINTS_DIR / f"{best_name}.pt", device="cpu")
    best_model = build_medical_classifier(best_meta["model_hparams"]).to(device).eval()
    best_model.load_state_dict(best_sd)
    drift_df = _simulate_clinical_drift_scenarios(
        best_model, best_meta, test_df, split_info, p_mal_cal_best, device, seed=seed
    )
    drift_df.to_csv(REPORTS_DIR / "drift_simulation_results.csv", index=False)

    # Build Master Comparison Table (W1-W5 Original + W6 Models + Published Benchmarks + Humans)
    master_rows = _read_original_domain_w1_w5_rows()
    for _, r in summary_df.iterrows():
        status = (
            "Best Medical Model"
            if r["run_name"] == best_name
            else (
                "Honest Negative (Frozen W2)"
                if "frozen" in str(r["run_name"])
                else "Medical Ablation / Transfer"
            )
        )
        master_rows.append(
            {
                "stage": "W6 (Medical)",
                "domain": "HAM10000 (Dermoscopy)",
                "model": str(r["run_name"]),
                "transfer_status": status,
                "primary_metric": (
                    f"BalAcc={r['balanced_accuracy']:.3f}, MalAUC={r['malignant_auc_roc']:.3f}, "
                    f"PR-AUC={r['malignant_pr_auc']:.3f}, Sens@Scr={r['malignant_sens_screening']:.3f}, "
                    f"Spec@Scr={r['malignant_spec_screening']:.3f}, ECE={r['ece_cal']:.3f}"
                ),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "malignant_auc_roc": float(r["malignant_auc_roc"]),
                "notes": f"Sens@Youden={r['malignant_sens_youden']:.3f}, PPV={r['malignant_ppv_youden']:.3f}, NPV={r['malignant_npv_youden']:.3f}",
            }
        )
    if vae_summary is not None:
        master_rows.append(
            {
                "stage": "W6 (Medical)",
                "domain": "HAM10000 (Unsup Anomaly)",
                "model": "w4_med_conv_vae2d",
                "transfer_status": "Unsupervised Anomaly/OOD Baseline",
                "primary_metric": (
                    f"MalAUC={vae_summary['auc_roc']:.3f}, PR-AUC={vae_summary['pr_auc']:.3f}, "
                    f"Sens={vae_summary['sensitivity']:.3f}, Spec={vae_summary['specificity']:.3f}"
                ),
                "balanced_accuracy": float(vae_summary["balanced_accuracy"]),
                "malignant_auc_roc": float(vae_summary["auc_roc"]),
                "notes": "Trained on benign lesions only; useful for OOD screening, weaker than supervised",
            }
        )
    master_rows.extend(PUBLISHED_BENCHMARKS)
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(REPORTS_DIR / "master_comparison_w1_w6.csv", index=False)

    # Save publication-ready figures
    _plot_roc_pr_curves(y_test_mal, roc_curves_map, FIGURES_DIR / "roc_pr_curves.png")
    _plot_calibration_and_dca(
        y_test_mal,
        p_mal_raw_best,
        p_mal_cal_best,
        dca_df,
        best_name,
        FIGURES_DIR / "calibration_dca_curves.png",
    )
    _plot_fairness_subgroups(fair_df, FIGURES_DIR / "fairness_subgroups.png")
    _plot_confusion_matrix(
        y_best,
        p_cal_best.argmax(axis=1),
        f"Normalized Test Confusion Matrix ({best_name})",
        FIGURES_DIR / "confusion_matrix_best.png",
    )
    _plot_ablation_comparison(summary_df, FIGURES_DIR / "ablation_comparison.png")

    print("\n=== Clinical Test Evaluation Complete ===")
    print(
        summary_df[
            [
                "run_name",
                "balanced_accuracy",
                "malignant_auc_roc",
                "malignant_pr_auc",
                "malignant_sens_screening",
                "ece_cal",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    return {"best_model": best_name, "n_models_evaluated": len(summary_df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Workshop 6 Reproducible Clinical Evaluation.")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    evaluate_all_checkpoints(device_str=args.device, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
