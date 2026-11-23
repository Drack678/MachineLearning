"""Pruebas unitarias y smoke test de Fase 4 (AE/VAE secuencial, free-bits y anomalías <5%)."""

from __future__ import annotations

import torch

from Workshop_4.src.autoencoder import (
    SeqTextAutoencoder,
    cyclical_kl_beta,
    kl_divergence,
    reconstruction_loss,
)
from Workshop_4.src.evaluate_ae import build_anomaly_benchmark


def test_seq_ae_shapes_and_shared_output_head() -> None:
    model = SeqTextAutoencoder(vocab_size=76, z_dim=128, variational=True, d_model=192)
    head_params = sum(p.numel() for p in model.out_head.parameters())
    assert head_params < 20_000, f"La cabeza de salida debe ser compartida (<20k), tiene {head_params}"

    x = torch.randint(0, 76, (4, 80))
    logits, mu, log_var = model(x)
    assert logits.shape == (4, 80, 76)
    assert mu.shape == (4, 128)
    assert log_var is not None and log_var.shape == (4, 128)


def test_free_bits_and_cyclical_kl_annealing() -> None:
    mu = torch.zeros(4, 32)
    log_var = torch.zeros(4, 32)
    assert float(kl_divergence(mu, log_var, free_bits=0.0).mean().item()) == 0.0
    assert float(kl_divergence(mu, log_var, free_bits=0.25).mean().item()) == 8.0

    b_start = cyclical_kl_beta(epoch=1, total_epochs=40, target_beta=4.0, n_cycles=4)
    b_mid = cyclical_kl_beta(epoch=6, total_epochs=40, target_beta=4.0, n_cycles=4)
    b_cycle2 = cyclical_kl_beta(epoch=11, total_epochs=40, target_beta=4.0, n_cycles=4)
    assert b_start == 0.0
    assert b_mid == 4.0
    assert b_cycle2 == 0.0


def test_anomaly_benchmark_ratio(synthetic_vocab) -> None:
    x_clean = torch.randint(1, synthetic_vocab["size"], (200, 80), dtype=torch.uint8)
    x_mix, y_anom = build_anomaly_benchmark(x_clean, synthetic_vocab, anomaly_ratio=0.05, seed=42)
    assert len(x_mix) == 200
    assert int(y_anom.sum()) == 10


def test_seq_vae_overfits_single_batch() -> None:
    g = torch.Generator().manual_seed(42)
    x = torch.randint(1, 20, (8, 80), generator=g)
    model = SeqTextAutoencoder(vocab_size=20, z_dim=64, variational=True, emb_dim=32, d_model=64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    for _ in range(140):
        opt.zero_grad(set_to_none=True)
        logits, mu, lv = model(x, sample_posterior=True)
        loss = reconstruction_loss(logits, x).mean() + 0.05 * kl_divergence(mu, lv, free_bits=0.25).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x, sample_posterior=False)
        acc = float((logits.argmax(-1) == x).float().mean().item())
    assert acc > 0.98, f"El VAE secuencial no logró overfit en 1 batch: acc={acc:.4f}"
