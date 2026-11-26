"""Unit and invariant tests for Workshop 6 preprocessing, lesion splits, imbalance, and DICOM PHI."""

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

from Workshop_6.src.config import CLASS_NAMES, DX_TO_ID, NUM_CLASSES  # noqa: E402
from Workshop_6.src.medical_preprocessing import (  # noqa: E402
    ClassBalancedFocalLoss,
    audit_dicom_phi_cleanliness,
    build_medical_transforms,
    compute_class_balanced_weights,
    create_synthetic_dermoscopy_dicom,
    crop_dermoscope_vignette,
    make_lesion_grouped_splits,
    preprocess_dermoscopy_image,
    read_dicom_medical_image,
    remove_hair_dullrazor,
    shades_of_gray_color_constancy,
    verify_lesion_splits,
)


def _mock_metadata_df(n_lesions: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    img_counter = 0
    for i in range(n_lesions):
        lesion_id = f"HAM_{i:05d}"
        dx = CLASS_NAMES[i % NUM_CLASSES]
        n_views = int(rng.integers(1, 4))
        for _ in range(n_views):
            rows.append(
                {
                    "lesion_id": lesion_id,
                    "image_id": f"ISIC_{img_counter:06d}",
                    "dx": dx,
                    "label": DX_TO_ID[dx],
                    "is_malignant": int(dx in ("akiec", "bcc", "mel")),
                    "dx_type": "histo",
                    "age": float(rng.integers(20, 85)),
                    "sex": "female" if i % 2 == 0 else "male",
                    "localization": "back",
                }
            )
            img_counter += 1
    return pd.DataFrame(rows)


def test_lesion_grouped_splits_have_zero_leakage() -> None:
    # Given: Synthetic HAM10000 metadata with multiple images per lesion_id.
    df = _mock_metadata_df(n_lesions=120)
    # When: Creating 70/15/15 grouped splits.
    splits = make_lesion_grouped_splits(df, seed=42)
    report = verify_lesion_splits(df, splits)
    # Then: Zero lesion_id and image_id overlap across all pairs.
    assert all(v == 0 for v in report["lesion_overlaps"].values())
    assert all(v == 0 for v in report["image_overlaps"].values())
    total_lesions = sum(report["counts"][k]["lesions"] for k in ("train", "val", "test"))
    assert total_lesions == 120


def test_dermoscopy_preprocessing_preserves_shape_and_removes_vignette() -> None:
    # Given: RGB image with dark circular-like vignette border and synthetic hair line.
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    img[20:140, 20:140] = np.array([190, 140, 120], dtype=np.uint8)
    img[40:120, 78:80] = 5  # Thin dark hair line
    # When: Running each stage and the full pipeline.
    cropped = crop_dermoscope_vignette(img)
    dehaired = remove_hair_dullrazor(cropped)
    cc = shades_of_gray_color_constancy(dehaired)
    out = preprocess_dermoscopy_image(img, image_size=64, enable_medical_preproc=True)
    # Then: Vignette is cropped, shapes are valid, and pixels stay uint8 [0, 255].
    assert cropped.shape[0] < img.shape[0] and cropped.shape[1] < img.shape[1]
    assert cc.dtype == np.uint8
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_medical_transforms_keep_hue_zero_and_tensor_shape() -> None:
    tf = build_medical_transforms(image_size=64, is_train=True)
    color_jitter = [t for t in tf.transforms if t.__class__.__name__ == "ColorJitter"]
    assert len(color_jitter) == 1
    assert color_jitter[0].hue is None or color_jitter[0].hue == (0.0, 0.0)


def test_class_balanced_focal_loss_is_finite_and_differentiable() -> None:
    counts = np.array([300, 500, 1000, 110, 1100, 6700, 140])
    weights = compute_class_balanced_weights(counts, beta=0.999)
    assert weights.shape == (NUM_CLASSES,)
    # Rare class (df=110) must receive higher weight than majority class (nv=6700)
    assert float(weights[3]) > float(weights[5])

    loss_fn = ClassBalancedFocalLoss(class_weights=weights, gamma=2.0)
    logits = torch.randn(16, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (16,))
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_dicom_reader_and_phi_audit_detect_synthetic_phi(tmp_path: Path) -> None:
    # Given: Two synthetic VL Dermoscopy DICOMs — one clean, one with synthetic PHI tags.
    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    clean_dcm = create_synthetic_dermoscopy_dicom(
        tmp_path / "clean.dcm", rgb, include_synthetic_phi=False
    )
    phi_dcm = create_synthetic_dermoscopy_dicom(
        tmp_path / "with_phi.dcm", rgb, include_synthetic_phi=True
    )

    # When: Reading pixel arrays and auditing PHI cleanliness.
    arr, tech = read_dicom_medical_image(clean_dcm)
    clean_audit = audit_dicom_phi_cleanliness(clean_dcm)
    phi_audit = audit_dicom_phi_cleanliness(phi_dcm)

    # Then: Pixel shape matches, clean file passes, PHI file flags tags without leaking values.
    assert arr.shape == (64, 64, 3) and tech["Modality"] == "XC"
    assert clean_audit["passes_basic_phi_check"] is True
    assert phi_audit["passes_basic_phi_check"] is False
    assert "PatientName" in phi_audit["phi_tags_present"]
    assert "SYNTHETIC" not in str(phi_audit)
