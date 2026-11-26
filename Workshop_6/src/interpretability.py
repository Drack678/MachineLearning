"""Clinical interpretability: Grad-CAM, Transformer Attention Rollout, anatomical alignment, and failure modes.

Implements Phase 4 of Workshop 6:
- `GradCAM` (Selvaraju et al., ICCV 2017) with deterministic hook cleanup for ResNet-18 and MedHybridTransformer.
- `AttentionRollout` (Abnar & Zuidema, ACL 2020) across the 4 W2-adapted TransformerEncoderLayer blocks.
- Quantitative anatomical validation against Tschandl et al. ground-truth lesion segmentation masks:
  Energy-in-Lesion Ratio, Pointing Game Hit Rate, and Saliency-Mask IoU@0.5.
- Qualitative 15-case test panel (5 TP, 5 FN, 5 FP) and structured 5-failure-mode clinical catalog.
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
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint  # noqa: E402
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything  # noqa: E402
from shared.metrics import apply_logit_adjustment  # noqa: E402
from Workshop_6.src.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    FIGURES_DIR,
    ID_TO_DX,
    REPORTS_DIR,
    SEED,
    XAI_FIGURES_DIR,
)
from Workshop_6.src.medical_metrics import malignant_probabilities  # noqa: E402
from Workshop_6.src.medical_preprocessing import (  # noqa: E402
    HAM10000Dataset,
    build_medical_transforms,
    get_cache_dir,
    load_split_dataframes,
    preprocess_dermoscopy_image,
)
from Workshop_6.src.transfer import (  # noqa: E402
    MedHybridTransformer,
    build_medical_classifier,
)

# ---------------------------------------------------------------------------
# Grad-CAM (Selvaraju et al., ICCV 2017)
# ---------------------------------------------------------------------------


class GradCAM:
    """Gradient-weighted Class Activation Mapping with guaranteed hook cleanup."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Compute normalized [0, 1] Grad-CAM heatmap for a single image tensor (1, C, H, W)."""
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"GradCAM expects shape (1, C, H, W), got {tuple(input_tensor.shape)}")

        self.model.eval()
        self._activations = None
        self._gradients = None

        def _fwd_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self._activations = out.detach()

        def _bwd_hook(_mod: nn.Module, _grad_in: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            self._gradients = grad_out[0].detach()

        h_fwd = self.target_layer.register_forward_hook(_fwd_hook)
        h_bwd = self.target_layer.register_full_backward_hook(_bwd_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x = input_tensor.detach().clone().requires_grad_(True)
                logits = self.model(x)
                cls_idx = (
                    int(logits.argmax(dim=-1).item()) if target_class is None else int(target_class)
                )
                score = logits[0, cls_idx]
                score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("GradCAM failed to capture activations or gradients.")

            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam[0, 0].cpu().numpy().astype(np.float32)
            c_min, c_max = float(cam_np.min()), float(cam_np.max())
            if c_max - c_min > 1e-8:
                cam_np = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_np = np.zeros_like(cam_np)
            return np.clip(cam_np, 0.0, 1.0), logits.detach().cpu()
        finally:
            h_fwd.remove()
            h_bwd.remove()
            self.model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Transformer Attention Rollout (Abnar & Zuidema, ACL 2020)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compute_attention_rollout(
    model: MedHybridTransformer,
    input_tensor: torch.Tensor,
    *,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> tuple[np.ndarray, torch.Tensor]:
    """Compute Attention Rollout heatmap over the 2D patch grid of MedHybridTransformer."""
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError(f"Expected shape (1, C, H, W), got {tuple(input_tensor.shape)}")
    model.eval()
    logits, attn_layers = model.forward_with_attention(input_tensor)

    n_tokens = attn_layers[0].shape[-1]
    eye = torch.eye(n_tokens, device=attn_layers[0].device, dtype=torch.float32)
    joint_attn = eye.clone()

    for attn in attn_layers:
        # attn: (1, n_heads, N, N)
        if head_fusion == "mean":
            fused = attn[0].float().mean(dim=0)
        elif head_fusion == "max":
            fused = attn[0].float().amax(dim=0)
        else:
            raise ValueError(f"Unsupported head_fusion: {head_fusion}")

        if discard_ratio > 0.0:
            flat = fused.view(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                thresh = torch.kthvalue(flat, k).values
                fused = torch.where(fused >= thresh, fused, torch.zeros_like(fused))

        # Residual connection: 0.5 * A + 0.5 * I, row-normalized
        aug = 0.5 * fused + 0.5 * eye
        aug = aug / aug.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint_attn = aug @ joint_attn

    # CLS token attention to spatial patches (excluding CLS self-attention at index 0)
    cls_attn = joint_attn[0, 1:]
    grid = cls_attn.reshape(1, 1, model.grid_size, model.grid_size)
    up = F.interpolate(
        grid,
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode="bilinear",
        align_corners=False,
    )
    arr = up[0, 0].cpu().numpy().astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if (hi - lo) > 1e-8 else np.zeros_like(arr)
    return np.clip(arr, 0.0, 1.0), logits.detach().cpu()


# ---------------------------------------------------------------------------
# Quantitative Anatomical Alignment Validation (vs Tschandl Segmentation Masks)
# ---------------------------------------------------------------------------


def compute_lesion_alignment_metrics(
    heatmap: np.ndarray, mask: np.ndarray | None, *, saliency_threshold: float = 0.5
) -> dict[str, float]:
    """Quantify whether saliency highlights clinically relevant lesion anatomy vs border shortcuts."""
    if mask is None or mask.size == 0 or float(mask.sum()) <= 0.0:
        return {
            "energy_in_lesion": float("nan"),
            "pointing_game_hit": float("nan"),
            "saliency_iou_05": float("nan"),
            "border_energy_ratio": float("nan"),
        }

    hmap = np.clip(np.asarray(heatmap, dtype=np.float64), 0.0, 1.0)
    m = np.asarray(mask, dtype=np.float64) > 0.5
    if m.shape != hmap.shape:
        m = (
            cv2.resize(
                m.astype(np.uint8), (hmap.shape[1], hmap.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            > 0
        )

    total_energy = float(hmap.sum())
    energy_in_lesion = float(hmap[m].sum() / total_energy) if total_energy > 1e-12 else 0.0

    peak_y, peak_x = np.unravel_index(int(np.argmax(hmap)), hmap.shape)
    pointing_hit = 1.0 if bool(m[peak_y, peak_x]) else 0.0

    sal_bin = hmap >= saliency_threshold
    inter = float(np.logical_and(sal_bin, m).sum())
    union = float(np.logical_or(sal_bin, m).sum())
    iou = inter / max(union, 1.0)

    # Outer 12% frame energy detects vignette/ruler shortcut learning
    h, w = hmap.shape
    bh, bw = max(1, int(0.12 * h)), max(1, int(0.12 * w))
    border_mask = np.ones_like(m, dtype=bool)
    border_mask[bh : h - bh, bw : w - bw] = False
    border_energy = float(hmap[border_mask].sum() / total_energy) if total_energy > 1e-12 else 0.0

    return {
        "energy_in_lesion": energy_in_lesion,
        "pointing_game_hit": pointing_hit,
        "saliency_iou_05": iou,
        "border_energy_ratio": border_energy,
    }


def overlay_heatmap_rgb(
    img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a colorblind-friendly magma/inferno heatmap onto an uint8 RGB image."""
    cmap = plt.get_cmap("inferno")
    colored = (cmap(np.clip(heatmap, 0.0, 1.0))[..., :3] * 255.0).astype(np.float32)
    base = img_rgb.astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 255.0)
    return blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# Qualitative 15-Case Suite (TP / FN / FP) & 5 Concrete Failure Modes
# ---------------------------------------------------------------------------


def _classify_failure_archetype(row: dict[str, Any]) -> tuple[str, str, str]:
    """Assign clinical failure mode category, patient harm, and clinician catch mechanism."""
    true_dx = str(row["true_dx"])
    pred_dx = str(row["pred_dx"])
    border_e = float(row.get("border_energy_ratio", 0.0))
    energy_in = float(row.get("energy_in_lesion", 1.0))
    age_grp = str(row.get("age_group", "unknown"))
    loc_grp = str(row.get("localization_group", "unknown"))

    if np.isfinite(border_e) and (
        border_e >= 0.30 or (np.isfinite(energy_in) and energy_in < 0.35)
    ):
        return (
            "Artifact / Border Shortcut Attention",
            "High: model relies on dermoscope rim, ruler markings, or residual hair rather than lesion morphology, risking silent failure on new clinics/scanners.",
            "Visualizing the Grad-CAM map immediately exposes off-lesion activation at image borders, prompting clinician override and standard dermoscopic inspection.",
        )
    if true_dx == "mel" and row["case_type"] == "FN":
        return (
            "Missed Melanoma (Hypopigmented / Atypical Mimic)",
            "Critical: delayed diagnosis of invasive melanoma substantially lowers 5-year survival and increases metastatic progression risk.",
            "Apply the clinical ABCDE/7-point checklist and patient history (evolving/new lesion), plus mandatory dermatologist review for any clinically suspicious lesion regardless of AI negative score.",
        )
    if true_dx in ("akiec", "bcc") and row["case_type"] == "FN":
        return (
            "Rare Non-Melanoma Malignancy False Negative",
            "Moderate-to-High: missed basal cell carcinoma or squamous precursor leads to local tissue invasion and more extensive surgical excision.",
            "Examine vascular arborizing patterns or scaling under polarized dermoscopy and flag elderly sun-exposed sites (head/neck) for heightened suspicion.",
        )
    if true_dx == "bkl" and row["case_type"] == "FP":
        return (
            "Pigmented Benign Keratosis Mimicking Melanoma (False Positive)",
            "Low-to-Moderate: induces patient anxiety, unnecessary excisional biopsy, scarring, and dermatology referral congestion.",
            "Dermatologist identifies classic seborrheic keratosis hallmarks (milia-like cysts, comedo-like openings, sharply demarcated borders) to safely defer biopsy.",
        )
    if age_grp in (">=75", "<40") or loc_grp in ("head_neck", "acral_special"):
        return (
            f"Subgroup Hidden Stratification ({age_grp} / {loc_grp})",
            "Moderate-to-High: systematic underperformance on underrepresented anatomical sites or extreme age brackets exacerbates healthcare disparities.",
            "Cross-check anatomical site-specific diagnostic criteria (e.g., parallel ridge pattern on acral skin) and consult model subgroup uncertainty warnings.",
        )
    return (
        f"Inter-class Morphological Overlap ({true_dx.upper()} vs {pred_dx.upper()})",
        "Moderate: incorrect malignancy stratification may alter triage urgency or biopsy decision.",
        "Correlate dermoscopic score with clinical lesion evolution, palpable induration, and multi-lesion 'ugly duckling' comparison.",
    )


def run_interpretability_analysis(
    *,
    primary_ckpt: Path,
    hybrid_ckpt: Path | None = None,
    device_str: str = "auto",
    n_per_group: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    """Generate 15-case TP/FN/FP visual panels, alignment statistics, and 5 failure modes."""
    seed_everything(seed, deterministic=True)
    device, desc = resolve_device(device_str)
    print(f"[XAI] Device: {device} | {desc}")

    sd, meta = load_checkpoint(primary_ckpt, device="cpu")
    model = build_medical_classifier(meta["model_hparams"]).to(device).eval()
    model.load_state_dict(sd)

    hybrid_model: MedHybridTransformer | None = None
    if hybrid_ckpt is not None and hybrid_ckpt.exists():
        h_sd, h_meta = load_checkpoint(hybrid_ckpt, device="cpu")
        if h_meta.get("model_hparams", {}).get("arch") == "MedHybridTransformer":
            hybrid_model = build_medical_classifier(h_meta["model_hparams"]).to(device).eval()  # type: ignore[assignment]
            hybrid_model.load_state_dict(h_sd)

    dfs, split_info = load_split_dataframes()
    test_df = dfs["test"].reset_index(drop=True)
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
        return_metadata=True,
        load_mask=True,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    cal = meta.get("calibration", {})
    temp = float(cal.get("temperature", 1.0))
    tau = float(cal.get("logit_adj_tau", 0.0))
    counts = np.asarray(cal.get("train_class_counts", [1] * len(CLASS_NAMES)))
    thr_bal = float(meta.get("clinical_thresholds_val", {}).get("balanced_youden", 0.5))

    logits_list: list[torch.Tensor] = []
    with torch.inference_mode():
        for xb, _, _ in loader:
            logits_list.append(model(xb.to(device)).float().cpu())
    logits_all = torch.cat(logits_list, dim=0)
    if tau > 0.0:
        logits_all = apply_logit_adjustment(logits_all, counts, tau)
    probs_all = torch.softmax(logits_all / temp, dim=-1).numpy()
    p_mal = malignant_probabilities(probs_all)
    y_mal = test_df["is_malignant"].to_numpy()
    pred_mal = (p_mal >= thr_bal).astype(np.int64)

    tp_idx = np.where((y_mal == 1) & (pred_mal == 1))[0]
    fn_idx = np.where((y_mal == 1) & (pred_mal == 0))[0]
    fp_idx = np.where((y_mal == 0) & (pred_mal == 1))[0]

    # Pick informative cases: confident TPs, worst FNs (lowest p_mal), worst FPs (highest p_mal)
    sel_tp = (
        tp_idx[np.argsort(-p_mal[tp_idx])[:n_per_group]] if len(tp_idx) else np.array([], dtype=int)
    )
    sel_fn = (
        fn_idx[np.argsort(p_mal[fn_idx])[:n_per_group]] if len(fn_idx) else np.array([], dtype=int)
    )
    sel_fp = (
        fp_idx[np.argsort(-p_mal[fp_idx])[:n_per_group]] if len(fp_idx) else np.array([], dtype=int)
    )

    selected: list[tuple[str, int]] = (
        [("TP", int(i)) for i in sel_tp]
        + [("FN", int(i)) for i in sel_fn]
        + [("FP", int(i)) for i in sel_fp]
    )

    XAI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model, model.get_last_conv_layer())  # type: ignore[attr-defined]
    case_records: list[dict[str, Any]] = []

    for rank, (case_type, idx) in enumerate(selected, start=1):
        x_t, y_true, info = ds[idx]
        x_in = x_t.unsqueeze(0).to(device)
        prob_vec = probs_all[idx]
        pred_cls = int(np.argmax(prob_vec))
        # Explain malignant peak class for malignant/FP cases or predicted class
        explain_cls = pred_cls if case_type != "FN" else int(y_true)
        cam_map, _ = gradcam.generate(x_in, target_class=explain_cls)

        mask_t = info.get("mask")
        mask_np = mask_t.numpy() if isinstance(mask_t, torch.Tensor) and mask_t.sum() > 0 else None
        align = compute_lesion_alignment_metrics(cam_map, mask_np)

        rollout_map: np.ndarray | None = None
        if hybrid_model is not None:
            rollout_map, _ = compute_attention_rollout(hybrid_model, x_in)
        elif isinstance(model, MedHybridTransformer):
            rollout_map, _ = compute_attention_rollout(model, x_in)

        raw_img = np.asarray(
            Image.open(test_df.iloc[idx]["image_path"]).convert("RGB"), dtype=np.uint8
        )
        vis_img = preprocess_dermoscopy_image(
            raw_img, image_size=img_sz, enable_medical_preproc=enable_pre
        )
        cam_overlay = overlay_heatmap_rgb(vis_img, cam_map)

        n_cols = 4 if rollout_map is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.6))
        axes[0].imshow(vis_img)
        if mask_np is not None:
            axes[0].contour(mask_np, levels=[0.5], colors=["#00FF00"], linewidths=1.2)
        axes[0].set_title(
            f"{case_type} #{rank}: {info['image_id']}\nTrue: {info['dx'].upper()} ({info['age_group']}, {info['sex']})",
            fontsize=8.5,
        )
        axes[0].axis("off")

        top3 = np.argsort(-prob_vec)[:3]
        bars_lbl = [ID_TO_DX[int(c)].upper() for c in top3]
        bars_val = [float(prob_vec[int(c)]) for c in top3]
        axes[1].barh(bars_lbl[::-1], bars_val[::-1], color="#0072B2")
        axes[1].set_xlim(0, 1.0)
        axes[1].set_title(
            f"Pred: {ID_TO_DX[pred_cls].upper()} | P(mal)={p_mal[idx]:.2f}\n(thr={thr_bal:.2f}, dx_type={info['dx_type']})",
            fontsize=8.5,
        )
        axes[1].grid(axis="x", alpha=0.3)

        axes[2].imshow(cam_overlay)
        axes[2].set_title(
            f"Grad-CAM ({ID_TO_DX[explain_cls].upper()})\nLesion E={align['energy_in_lesion']:.2f}, IoU={align['saliency_iou_05']:.2f}",
            fontsize=8.5,
        )
        axes[2].axis("off")

        if rollout_map is not None:
            axes[3].imshow(overlay_heatmap_rgb(vis_img, rollout_map))
            r_align = compute_lesion_alignment_metrics(rollout_map, mask_np)
            axes[3].set_title(
                f"W2-Hybrid Attn Rollout\nLesion E={r_align['energy_in_lesion']:.2f}",
                fontsize=8.5,
            )
            axes[3].axis("off")

        fig.tight_layout()
        panel_path = XAI_FIGURES_DIR / f"case_{rank:02d}_{case_type}_{info['image_id']}.png"
        fig.savefig(panel_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        case_records.append(
            {
                "case_rank": rank,
                "case_type": case_type,
                "image_id": info["image_id"],
                "lesion_id": info["lesion_id"],
                "true_dx": info["dx"],
                "pred_dx": ID_TO_DX[pred_cls],
                "p_malignant": float(p_mal[idx]),
                "age_group": info["age_group"],
                "sex": info["sex"],
                "localization_group": info["localization_group"],
                "dx_type": info["dx_type"],
                **align,
                "panel_file": panel_path.name,
            }
        )

    cases_df = pd.DataFrame(case_records)
    cases_df.to_csv(REPORTS_DIR / "interpretability_15_cases.csv", index=False)

    # Select 5 distinct concrete failure modes from FN & FP cases
    failures = [r for r in case_records if r["case_type"] in ("FN", "FP")]
    catalog: list[dict[str, Any]] = []
    seen_archetypes: set[str] = set()
    for rec in failures:
        arch_name, harm, catch = _classify_failure_archetype(rec)
        if arch_name not in seen_archetypes or len(catalog) < 5:
            seen_archetypes.add(arch_name)
            catalog.append(
                {
                    "failure_mode_id": len(catalog) + 1,
                    "archetype": arch_name,
                    "case_type": rec["case_type"],
                    "image_id": rec["image_id"],
                    "true_dx": rec["true_dx"],
                    "pred_dx": rec["pred_dx"],
                    "p_malignant": rec["p_malignant"],
                    "age_group": rec["age_group"],
                    "sex": rec["sex"],
                    "localization_group": rec["localization_group"],
                    "dx_type": rec["dx_type"],
                    "energy_in_lesion": rec["energy_in_lesion"],
                    "border_energy_ratio": rec["border_energy_ratio"],
                    "patient_harm": harm,
                    "clinician_detection_strategy": catch,
                    "panel_file": rec["panel_file"],
                }
            )
        if len(catalog) >= 5:
            break

    (REPORTS_DIR / "failure_modes_catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    _save_failure_modes_grid(catalog, test_df, model, gradcam, ds, device, img_sz, enable_pre)

    # Quantitative summary across TP / FN / FP
    summary = {
        "primary_model": meta.get("run_name", primary_ckpt.stem),
        "n_cases_evaluated": len(case_records),
        "mean_energy_in_lesion": float(np.nanmean(cases_df["energy_in_lesion"]))
        if len(cases_df)
        else float("nan"),
        "pointing_game_accuracy": float(np.nanmean(cases_df["pointing_game_hit"]))
        if len(cases_df)
        else float("nan"),
        "mean_saliency_iou_05": float(np.nanmean(cases_df["saliency_iou_05"]))
        if len(cases_df)
        else float("nan"),
        "mean_border_energy_ratio": float(np.nanmean(cases_df["border_energy_ratio"]))
        if len(cases_df)
        else float("nan"),
        "failure_modes_documented": len(catalog),
    }
    (REPORTS_DIR / "interpretability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[XAI OK]", json.dumps(summary, indent=2))
    return summary


def _save_failure_modes_grid(
    catalog: list[dict[str, Any]],
    test_df: pd.DataFrame,
    model: nn.Module,
    gradcam: GradCAM,
    ds: HAM10000Dataset,
    device: torch.device,
    img_sz: int,
    enable_pre: bool,
) -> None:
    if not catalog:
        return
    id_to_row = {str(r["image_id"]): idx for idx, r in test_df.iterrows()}
    n = len(catalog)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 2.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, fm in enumerate(catalog):
        ds_idx = id_to_row[fm["image_id"]]
        x_t, y_true, _ = ds[ds_idx]
        cam, _ = gradcam.generate(x_t.unsqueeze(0).to(device), target_class=int(y_true))
        raw = np.asarray(
            Image.open(test_df.iloc[ds_idx]["image_path"]).convert("RGB"), dtype=np.uint8
        )
        vis = preprocess_dermoscopy_image(raw, image_size=img_sz, enable_medical_preproc=enable_pre)
        axes[r_idx, 0].imshow(vis)
        axes[r_idx, 0].set_title(
            f"FM{fm['failure_mode_id']} ({fm['case_type']}): {fm['archetype']}\n"
            f"{fm['image_id']} | True={fm['true_dx'].upper()} Pred={fm['pred_dx'].upper()} P(mal)={fm['p_malignant']:.2f}",
            fontsize=8,
        )
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(overlay_heatmap_rgb(vis, cam))
        axes[r_idx, 1].set_title(
            f"Grad-CAM | Lesion E={fm['energy_in_lesion']:.2f} | Border E={fm['border_energy_ratio']:.2f}",
            fontsize=8,
        )
        axes[r_idx, 1].axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_modes_grid.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workshop 6 Clinical Interpretability & Failure Modes."
    )
    parser.add_argument(
        "--primary-ckpt",
        type=str,
        default=str(CHECKPOINTS_DIR / "a1_resnet18_imagenet_finetune.pt"),
    )
    parser.add_argument(
        "--hybrid-ckpt",
        type=str,
        default=str(CHECKPOINTS_DIR / "a1_med_hybrid_w2_finetune.pt"),
    )
    parser.add_argument("--n-per-group", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_interpretability_analysis(
        primary_ckpt=Path(args.primary_ckpt),
        hybrid_ckpt=Path(args.hybrid_ckpt) if args.hybrid_ckpt else None,
        device_str=args.device,
        n_per_group=args.n_per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
