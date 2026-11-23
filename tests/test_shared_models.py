"""Pruebas de contratos de CharTransformer y persistencia segura (weights_only=True)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shared.checkpoints import load_checkpoint, save_checkpoint
from shared.models import CharTransformer, build_warmup_cosine_scheduler, count_parameters


def test_char_transformer_shapes_and_weight_tying() -> None:
    model = CharTransformer(
        vocab_size=76, d_model=64, n_heads=4, n_layers=2, dropout=0.0, tie_weights=True
    )
    model.eval()
    x = torch.randint(0, 76, (4, 80))

    logits_last = model(x, return_all_tokens=False)
    logits_all = model(x, return_all_tokens=True)
    feats = model.features(x)

    assert logits_last.shape == (4, 76)
    assert logits_all.shape == (4, 80, 76)
    assert feats.shape == (4, 64)
    assert torch.allclose(logits_last, logits_all[:, -1], atol=1e-5)
    assert model.fc.weight is model.token_embedding.weight
    assert count_parameters(model) > 0


def test_char_transformer_strict_causality() -> None:
    model = CharTransformer(vocab_size=32, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model.eval()
    x1 = torch.randint(0, 32, (2, 20))
    x2 = x1.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 32

    with torch.no_grad():
        out1 = model(x1, return_all_tokens=True)
        out2 = model(x2, return_all_tokens=True)

    assert torch.allclose(out1[:, :10], out2[:, :10], atol=1e-5)
    assert not torch.allclose(out1[:, 10:], out2[:, 10:])


def test_safe_checkpoint_roundtrip_weights_only(tmp_path: Path) -> None:
    model = CharTransformer(vocab_size=32, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(
        ckpt_path,
        model.state_dict(),
        metadata={"model_hparams": model.hparams, "epoch": 3, "val_bpc": 1.42},
    )

    state_dict, meta = load_checkpoint(ckpt_path, device="cpu")
    restored = CharTransformer(**meta["model_hparams"])
    restored.load_state_dict(state_dict)

    assert meta["epoch"] == 3
    assert meta["val_bpc"] == pytest.approx(1.42)
    x = torch.randint(0, 32, (2, 16))
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x))


def test_warmup_cosine_scheduler_curve() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = build_warmup_cosine_scheduler(opt, total_steps=100, warmup_ratio=0.1, min_lr_ratio=0.01)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]
    assert lrs[9] == pytest.approx(1e-3, rel=1e-3)
    assert lrs[-1] < 2e-5
