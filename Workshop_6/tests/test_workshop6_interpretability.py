"""Unit tests for Grad-CAM, Attention Rollout, hook cleanup, and lesion anatomical alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_6.src.config import NUM_CLASSES  # noqa: E402
from Workshop_6.src.interpretability import (  # noqa: E402
    GradCAM,
    compute_attention_rollout,
    compute_lesion_alignment_metrics,
)
from Workshop_6.src.transfer import MedHybridTransformer, ResNet18Medical  # noqa: E402


def test_gradcam_produces_finite_normalized_map_and_removes_hooks() -> None:
    model = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=False).eval()
    target_layer = model.get_last_conv_layer()
    cam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 64, 64)

    heatmap, logits = cam.generate(x, target_class=4)
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0 and float(heatmap.max()) <= 1.0
    assert logits.shape == (1, NUM_CLASSES)
    # Verify deterministic hook cleanup (zero lingering hooks)
    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0


def test_attention_rollout_and_gradcam_on_med_hybrid_transformer() -> None:
    model = MedHybridTransformer(
        num_classes=NUM_CLASSES, image_size=64, d_model=64, n_heads=4, n_layers=2
    ).eval()
    x = torch.randn(1, 3, 64, 64)

    rollout, logits_r = compute_attention_rollout(model, x)
    cam = GradCAM(model, model.get_last_conv_layer())
    heatmap, logits_c = cam.generate(x, target_class=4)

    assert rollout.shape == (64, 64) and np.isfinite(rollout).all()
    assert float(rollout.min()) >= 0.0 and float(rollout.max()) <= 1.0
    assert heatmap.shape == (64, 64) and np.isfinite(heatmap).all()
    assert torch.allclose(logits_r, logits_c, atol=1e-4)


def test_lesion_alignment_metrics_perfect_vs_border_shortcut() -> None:
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:44, 20:44] = 1.0

    # Ideal saliency inside lesion
    good_map = mask.copy()
    m_good = compute_lesion_alignment_metrics(good_map, mask)
    assert np.isclose(m_good["energy_in_lesion"], 1.0)
    assert m_good["pointing_game_hit"] == 1.0
    assert np.isclose(m_good["saliency_iou_05"], 1.0)
    assert np.isclose(m_good["border_energy_ratio"], 0.0)

    # Shortcut saliency strictly on top border
    bad_map = np.zeros((64, 64), dtype=np.float32)
    bad_map[0:5, :] = 1.0
    m_bad = compute_lesion_alignment_metrics(bad_map, mask)
    assert np.isclose(m_bad["energy_in_lesion"], 0.0)
    assert m_bad["pointing_game_hit"] == 0.0
    assert np.isclose(m_bad["border_energy_ratio"], 1.0)
