"""Auto-generate LaTeX metric macros and booktabs tables from verified W1-W6 CSV/JSON outputs.

Never fabricates numbers: if a Workshop 6 run has not yet produced a CSV/JSON artifact
(e.g., before running `scripts/run_all_rocm.sh` on AMD RX 9060 XT), emits explicit
`[PENDING ROCM RUN]` markers so the LaTeX manuscript remains honest and verifiable.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_6.src.config import LATEX_DIR, REPORTS_DIR, SPLITS_FILE  # noqa: E402


def _tex_escape(text: Any) -> str:
    s = str(text)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        s = s.replace(old, new)
    return s


def _fmt(val: Any, digits: int = 3, pct: bool = False) -> str:
    try:
        v = float(val)
        if not math.isfinite(v):
            return "--"
        if pct:
            return f"{v * 100.0:.1f}\\%"
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _write_booktabs_table(
    headers: list[str],
    rows: list[list[str]],
    col_spec: str,
    out_path: Path,
) -> None:
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_latex_assets() -> dict[str, Any]:
    tables_dir = LATEX_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}

    # 1. Original-domain verified metrics (W1-W5)
    w1_csv = REPO_ROOT / "Workshop_1" / "reports" / "baselines_summary.csv"
    if w1_csv.exists():
        df1 = pd.read_csv(w1_csv).set_index("model")
        macros["WOneLogAcc"] = _fmt(df1.loc["logistic_sgd_balanced", "test_accuracy"], pct=True)
        macros["WOneLogFOne"] = _fmt(df1.loc["logistic_sgd_balanced", "test_f1_macro"], 4)
        macros["WOneLogBPC"] = _fmt(df1.loc["logistic_sgd_balanced", "test_bpc"], 3)
        macros["WOneNBAcc"] = _fmt(df1.loc["complement_nb", "test_accuracy"], pct=True)
        macros["WOneNBFOne"] = _fmt(df1.loc["complement_nb", "test_f1_macro"], 4)

    w2_csv = REPO_ROOT / "Workshop_2" / "reports" / "results_comparison.csv"
    if w2_csv.exists():
        df2 = pd.read_csv(w2_csv)
        cal = df2[(df2["split"] == "test_full") & (df2["model"].str.contains("calibrated"))].iloc[0]
        adj = df2[(df2["split"] == "test_full") & (df2["model"].str.contains("logit-adj"))].iloc[0]
        macros["WTwoCalAcc"] = _fmt(cal["accuracy"], pct=True)
        macros["WTwoCalTopFive"] = _fmt(cal["top5_accuracy"], pct=True)
        macros["WTwoCalBPC"] = _fmt(cal["bpc"], 4)
        macros["WTwoCalFOne"] = _fmt(cal["f1_macro"], 4)
        macros["WTwoCalECE"] = _fmt(cal["ece"], 4)
        macros["WTwoCalAUC"] = _fmt(cal["auc_roc_ovr_macro"], 4)
        macros["WTwoAdjFOne"] = _fmt(adj["f1_macro"], 4)
        macros["WTwoAdjECE"] = _fmt(adj["ece"], 4)

    w3_csv = REPO_ROOT / "Workshop_3" / "reports" / "test_comparison.csv"
    if w3_csv.exists():
        df3 = pd.read_csv(w3_csv)
        dqn = df3[df3["policy"].str.contains("DQN", case=False)].iloc[0]
        greedy = df3[df3["policy"].str.contains("greedy", case=False)].iloc[0]
        macros["WThreeDQNReturn"] = f"{float(dqn['mean_return']):+.3f}"
        macros["WThreeDQNNewWord"] = _fmt(dqn["new_word_rate"], pct=True)
        macros["WThreeDQNInvalid"] = _fmt(dqn["invalid_word_rate"], pct=True)
        macros["WThreeGreedyReturn"] = f"{float(greedy['mean_return']):+.3f}"

    w4_rec = REPO_ROOT / "Workshop_4" / "reports" / "reconstruction_metrics.csv"
    w4_anom = REPO_ROOT / "Workshop_4" / "reports" / "anomaly_detection_metrics.csv"
    if w4_rec.exists() and w4_anom.exists():
        r4 = pd.read_csv(w4_rec)
        a4 = pd.read_csv(w4_anom)
        ae_r = r4[r4["model"].str.startswith("AE")].iloc[0]
        vae_r = r4[r4["model"].str.startswith("VAE")].iloc[0]
        vae_a = a4[a4["method"].str.contains("recon \\+ KL")].iloc[0]
        macros["WFourAEAcc"] = _fmt(ae_r["char_accuracy"], pct=True)
        macros["WFourVAEAcc"] = _fmt(vae_r["char_accuracy"], pct=True)
        macros["WFourVAEAnomFOne"] = _fmt(vae_a["f1"], 4)

    w5_aug = REPO_ROOT / "Workshop_5" / "reports" / "metrics" / "augmentation_results.csv"
    w5_qual = REPO_ROOT / "Workshop_5" / "reports" / "metrics" / "gan_quality.csv"
    if w5_aug.exists() and w5_qual.exists():
        d5a = pd.read_csv(w5_aug).set_index("condition")
        d5q = pd.read_csv(w5_qual).set_index("model")
        macros["WFiveRealBPC"] = _fmt(d5a.loc["real_only", "test_bpc"], 4)
        macros["WFiveWGANBPC"] = _fmt(d5a.loc["wgan_gp_filtered", "test_bpc"], 4)
        macros["WFiveNoiseFOne"] = _fmt(d5a.loc["char_noise_aug", "test_f1_macro_cal"], 4)
        macros["WFiveVanillaPPL"] = _fmt(d5q.loc["vanilla", "lm_perplexity"], 2)
        macros["WFiveWGANPPL"] = _fmt(d5q.loc["wgan_gp", "lm_perplexity"], 1)

    # 2. Workshop 6 Splits & Dataset Metadata
    if SPLITS_FILE.exists():
        sp = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
        cnt = sp["verification"]["counts"]
        macros["WSixTrainImgs"] = f"{cnt['train']['images']:,}"
        macros["WSixValImgs"] = f"{cnt['val']['images']:,}"
        macros["WSixTestImgs"] = f"{cnt['test']['images']:,}"
        macros["WSixTrainLesions"] = f"{cnt['train']['lesions']:,}"
        macros["WSixValLesions"] = f"{cnt['val']['lesions']:,}"
        macros["WSixTestLesions"] = f"{cnt['test']['lesions']:,}"
    else:
        for k in (
            "WSixTrainImgs",
            "WSixValImgs",
            "WSixTestImgs",
            "WSixTrainLesions",
            "WSixValLesions",
            "WSixTestLesions",
        ):
            macros[k] = "[PENDING]"

    # 3. Workshop 6 Clinical Evaluation & Ablations Table
    clin_csv = REPORTS_DIR / "clinical_metrics_summary.csv"
    if clin_csv.exists():
        df_c = pd.read_csv(clin_csv)
        best = df_c.iloc[0]
        macros["WSixBestModel"] = _tex_escape(best["run_name"])
        macros["WSixBestBalAcc"] = _fmt(best["balanced_accuracy"], 3)
        macros["WSixBestMalAUC"] = _fmt(best["malignant_auc_roc"], 3)
        macros["WSixBestMalPRAUC"] = _fmt(best["malignant_pr_auc"], 3)
        macros["WSixBestSensYouden"] = _fmt(best["malignant_sens_youden"], pct=True)
        macros["WSixBestSpecYouden"] = _fmt(best["malignant_spec_youden"], pct=True)
        macros["WSixBestSensScr"] = _fmt(best["malignant_sens_screening"], pct=True)
        macros["WSixBestSpecScr"] = _fmt(best["malignant_spec_screening"], pct=True)
        macros["WSixBestECE"] = _fmt(best["ece_cal"], 4)

        ab_rows = [
            [
                _tex_escape(r["run_name"]),
                f"{int(r['params']) / 1e6:.2f}M",
                _fmt(r["balanced_accuracy"], 3),
                _fmt(r["f1_macro"], 3),
                _fmt(r["malignant_auc_roc"], 3),
                _fmt(r["malignant_pr_auc"], 3),
                _fmt(r["malignant_sens_screening"], pct=True),
                _fmt(r["ece_cal"], 3),
            ]
            for _, r in df_c.iterrows()
        ]
        _write_booktabs_table(
            [
                "Model / Ablation",
                "Params",
                "Bal. Acc",
                "Macro-F1",
                "Mal. AUC",
                "PR-AUC",
                "Sens@Scr",
                "ECE",
            ],
            ab_rows,
            "lrrrrrrr",
            tables_dir / "tab_ablations.tex",
        )
    else:
        for k in (
            "WSixBestModel",
            "WSixBestBalAcc",
            "WSixBestMalAUC",
            "WSixBestMalPRAUC",
            "WSixBestSensYouden",
            "WSixBestSpecYouden",
            "WSixBestSensScr",
            "WSixBestSpecScr",
            "WSixBestECE",
        ):
            macros[k] = "[PENDING]"
        _write_booktabs_table(
            [
                "Model / Ablation",
                "Params",
                "Bal. Acc",
                "Macro-F1",
                "Mal. AUC",
                "PR-AUC",
                "Sens@Scr",
                "ECE",
            ],
            [
                [
                    r"\multicolumn{8}{c}{\textit{Run \texttt{scripts/run\_all\_rocm.sh} to populate test metrics.}}"
                ]
            ],
            "lrrrrrrr",
            tables_dir / "tab_ablations.tex",
        )

    # 4. Threshold Analysis Table (Best Model)
    thr_csv = REPORTS_DIR / "threshold_analysis.csv"
    if thr_csv.exists() and clin_csv.exists():
        df_t = pd.read_csv(thr_csv)
        best_run = str(pd.read_csv(clin_csv).iloc[0]["run_name"])
        sub_t = df_t[df_t["run_name"] == best_run]
        t_rows = [
            [
                _tex_escape(r["operating_point"]),
                _fmt(r["threshold"], 3),
                f"{_fmt(r['sensitivity'], pct=True)} [{_fmt(r['sensitivity_ci_low'], pct=True)}--{_fmt(r['sensitivity_ci_high'], pct=True)}]",
                f"{_fmt(r['specificity'], pct=True)} [{_fmt(r['specificity_ci_low'], pct=True)}--{_fmt(r['specificity_ci_high'], pct=True)}]",
                _fmt(r["ppv"], pct=True),
                _fmt(r["npv"], pct=True),
                _fmt(r["fn_per_1000"], 1),
            ]
            for _, r in sub_t.iterrows()
        ]
        _write_booktabs_table(
            [
                "Operating Point",
                "Thr.",
                "Sensitivity (95\\% CI)",
                "Specificity (95\\% CI)",
                "PPV",
                "NPV",
                "FN/1k",
            ],
            t_rows,
            "lrrrrrr",
            tables_dir / "tab_thresholds.tex",
        )
    else:
        _write_booktabs_table(
            ["Operating Point", "Thr.", "Sensitivity", "Specificity", "PPV", "NPV", "FN/1k"],
            [[r"\multicolumn{7}{c}{\textit{Pending ROCm evaluation run.}}"]],
            "lrrrrrr",
            tables_dir / "tab_thresholds.tex",
        )

    # 5. Subgroup Fairness Table
    fair_csv = REPORTS_DIR / "subgroup_fairness_audit.csv"
    if fair_csv.exists():
        df_f = pd.read_csv(fair_csv)
        f_rows = [
            [
                _tex_escape(r["attribute"]),
                _tex_escape(r["subgroup"]),
                str(int(r["n_samples"])),
                _fmt(r["sensitivity"], pct=True),
                _fmt(r["specificity"], pct=True),
                _fmt(r["auc_roc"], 3),
                "ALERT" if bool(r["hidden_stratification_alert"]) else "OK",
            ]
            for _, r in df_f.iterrows()
        ]
        _write_booktabs_table(
            ["Attribute", "Subgroup", "N", "Sens.", "Spec.", "AUC", "Status"],
            f_rows,
            "llrrrrl",
            tables_dir / "tab_fairness.tex",
        )
    else:
        _write_booktabs_table(
            ["Attribute", "Subgroup", "N", "Sens.", "Spec.", "AUC", "Status"],
            [[r"\multicolumn{7}{c}{\textit{Pending ROCm evaluation run.}}"]],
            "llrrrrl",
            tables_dir / "tab_fairness.tex",
        )

    # 6. Drift Simulation Table
    drift_csv = REPORTS_DIR / "drift_simulation_results.csv"
    if drift_csv.exists():
        df_d = pd.read_csv(drift_csv)
        d_rows = [
            [
                _tex_escape(r["scenario"]),
                _fmt(r["pred_psi"], 3),
                _fmt(r["pred_ks_stat"], 3),
                _fmt(r["malignant_auc"], 3),
                _fmt(r["malignant_sensitivity"], pct=True),
                _tex_escape(r["recommended_action"]),
            ]
            for _, r in df_d.iterrows()
        ]
        _write_booktabs_table(
            ["Clinical Shift Scenario", "Pred PSI", "KS Stat", "Mal. AUC", "Sens.", "Trigger"],
            d_rows,
            "lrrrrl",
            tables_dir / "tab_drift.tex",
        )
    else:
        _write_booktabs_table(
            ["Clinical Shift Scenario", "Pred PSI", "KS Stat", "Mal. AUC", "Sens.", "Trigger"],
            [[r"\multicolumn{6}{c}{\textit{Pending ROCm evaluation run.}}"]],
            "lrrrrl",
            tables_dir / "tab_drift.tex",
        )

    # 7. Master W1-W6 Comparison Table
    master_csv = REPORTS_DIR / "master_comparison_w1_w6.csv"
    if master_csv.exists():
        df_m = pd.read_csv(master_csv)
        m_rows = [
            [
                _tex_escape(r["stage"]),
                _tex_escape(r["model"]),
                _tex_escape(r["transfer_status"]),
                _tex_escape(r["primary_metric"]),
            ]
            for _, r in df_m.iterrows()
        ]
        _write_booktabs_table(
            ["Stage", "Model / Benchmark", "Transfer Verdict", "Key Reported Metrics"],
            m_rows,
            r"p{2.2cm}p{4.1cm}p{3.0cm}p{5.5cm}",
            tables_dir / "tab_master_w1_w6.tex",
        )
    else:
        _write_booktabs_table(
            ["Stage", "Model / Benchmark", "Transfer Verdict", "Key Reported Metrics"],
            [[r"\multicolumn{4}{c}{\textit{Pending ROCm evaluation run.}}"]],
            r"p{2.2cm}p{4.1cm}p{3.0cm}p{5.5cm}",
            tables_dir / "tab_master_w1_w6.tex",
        )

    # 8. Interpretability Summary & 5 Failure Modes
    xai_json = REPORTS_DIR / "interpretability_summary.json"
    fm_json = REPORTS_DIR / "failure_modes_catalog.json"
    if xai_json.exists():
        xs = json.loads(xai_json.read_text(encoding="utf-8"))
        macros["WSixXaiLesionEnergy"] = _fmt(xs.get("mean_energy_in_lesion"), pct=True)
        macros["WSixXaiPointingAcc"] = _fmt(xs.get("pointing_game_accuracy"), pct=True)
        macros["WSixXaiIoU"] = _fmt(xs.get("mean_saliency_iou_05"), 3)
        macros["WSixXaiBorderEnergy"] = _fmt(xs.get("mean_border_energy_ratio"), pct=True)
    else:
        for k in ("WSixXaiLesionEnergy", "WSixXaiPointingAcc", "WSixXaiIoU", "WSixXaiBorderEnergy"):
            macros[k] = "[PENDING]"

    if fm_json.exists():
        fms = json.loads(fm_json.read_text(encoding="utf-8"))
        fm_rows = [
            [
                f"FM{fm['failure_mode_id']} ({_tex_escape(fm['case_type'])})",
                _tex_escape(fm["image_id"]),
                f"{_tex_escape(fm['true_dx']).upper()} $\\to$ {_tex_escape(fm['pred_dx']).upper()} ({_fmt(fm['p_malignant'], 2)})",
                _tex_escape(fm["archetype"]),
                _tex_escape(fm["clinician_detection_strategy"]),
            ]
            for fm in fms
        ]
        _write_booktabs_table(
            ["Mode", "Case ID", "Error (P(mal))", "Root Archetype", "Clinician Catch Mechanism"],
            fm_rows,
            r"p{1.6cm}p{2.1cm}p{2.3cm}p{3.3cm}p{5.3cm}",
            tables_dir / "tab_failure_modes.tex",
        )
    else:
        _write_booktabs_table(
            ["Mode", "Case ID", "Error (P(mal))", "Root Archetype", "Clinician Catch Mechanism"],
            [[r"\multicolumn{5}{c}{\textit{Pending ROCm interpretability run.}}"]],
            r"p{1.6cm}p{2.1cm}p{2.3cm}p{3.3cm}p{5.3cm}",
            tables_dir / "tab_failure_modes.tex",
        )

    # Write generated_metrics.tex
    macro_lines = [
        "% Auto-generated by Workshop_6/src/build_report_assets.py — DO NOT EDIT BY HAND",
        *[f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())],
        "",
    ]
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / "generated_metrics.tex").write_text("\n".join(macro_lines), encoding="utf-8")
    return macros


if __name__ == "__main__":
    out = generate_latex_assets()
    print(f"[LaTeX Assets] Generated {len(out)} macros and tables in {LATEX_DIR}")
