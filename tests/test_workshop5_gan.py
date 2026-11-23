"""Pruebas unitarias de Fase 5 (GAN 1D Gumbel-ST, WGAN-GP gradient penalty y augmentation)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Workshop_5.src.augmentation_and_retrain import (
    build_char_noise_augmented_blocks,
    train_condition,
)
from Workshop_5.src.gan import (
    SEQUENCE_LENGTH,
    Critic,
    Discriminator,
    Generator,
    gumbel_temperature,
)
from Workshop_5.src.train_wgan import gradient_penalty


def test_generator_gumbel_straight_through_is_discrete_and_differentiable() -> None:
    gen = Generator(vocab_size=32, noise_dim=32, hidden_dim=64)
    disc = Discriminator(vocab_size=32, hidden_dim=64)

    noise = torch.randn(4, 32)
    samples = gen(noise, tau=0.8, hard=True)
    assert samples.shape == (4, SEQUENCE_LENGTH, 32)
    assert torch.all((samples == 0.0) | (samples == 1.0))
    assert torch.allclose(samples.sum(dim=-1), torch.ones(4, SEQUENCE_LENGTH))

    score = disc(samples).mean()
    score.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in gen.parameters() if p.grad is not None)
    assert grad_sum > 0.0


def test_wgan_gradient_penalty_finite_and_differentiable() -> None:
    critic = Critic(vocab_size=32, hidden_dim=64)
    real = F.one_hot(torch.randint(0, 32, (4, SEQUENCE_LENGTH)), num_classes=32).float()
    fake = F.one_hot(torch.randint(0, 32, (4, SEQUENCE_LENGTH)), num_classes=32).float()
    gp = gradient_penalty(critic, real, fake, torch.device("cpu"))
    assert torch.isfinite(gp)
    gp.backward()
    assert gumbel_temperature(1, 20, 2.0, 0.5) == 2.0
    assert gumbel_temperature(20, 20, 2.0, 0.5) == 0.5


def test_char_noise_aug_preserves_targets_and_train_condition_runs(tmp_path: Path, monkeypatch) -> None:
    import Workshop_5.src.augmentation_and_retrain as aug_mod

    monkeypatch.setattr(aug_mod, "OUTPUT_CHECKPOINT_DIR", tmp_path / "ckpts")
    monkeypatch.setattr(aug_mod, "RUNS_DIR", tmp_path / "runs")

    g = torch.Generator().manual_seed(42)
    blocks = torch.randint(1, 20, (16, SEQUENCE_LENGTH), generator=g, dtype=torch.uint8)
    probs = np.full(20, 1.0 / 20.0)

    x_aug, y_aug = build_char_noise_augmented_blocks(blocks, probs, n_aug=8, noise_prob=0.1, seed=42)
    assert x_aug.shape == (8, 80)
    assert y_aug.shape == (8, 80)

    x_val = blocks[:8, :-1]
    y_val = blocks[:8, -1]
    counts = np.ones(20, dtype=np.int64) * 10
    args = argparse.Namespace(
        seed=42, profile="cpu", batch_size=8, epochs=2, learning_rate=1e-2,
        weight_decay=0.0, patience=2, num_workers=0,
    )
    hparams = {
        "vocab_size": 20, "d_model": 32, "n_heads": 4, "n_layers": 1,
        "dropout": 0.0, "max_len": 80, "activation": "gelu", "tie_weights": True,
    }
    res, hist = train_condition(
        "smoke", blocks[:, :-1], blocks[:, 1:], x_val, y_val, x_val, y_val,
        hparams, counts, args, torch.device("cpu"),
    )
    assert len(hist) == 2
    assert np.isfinite(res["test_bpc"])
    assert (tmp_path / "ckpts" / "transformer_smoke.pt").exists()
    assert (tmp_path / "ckpts" / "transformer_smoke.meta.json").exists()
