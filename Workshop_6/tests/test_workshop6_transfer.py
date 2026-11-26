"""Unit and smoke tests for Workshop 6 transfer models (W2 Hybrid, ResNet-18, W4 VAE, W5 WGAN-GP)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

W6_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = W6_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from shared.device import cpu_state_dict  # noqa: E402
from shared.models import CharTransformer  # noqa: E402
from Workshop_6.src.config import NUM_CLASSES  # noqa: E402
from Workshop_6.src.transfer import (  # noqa: E402
    CondCritic2D,
    CondGen2D,
    MedConvVAE2D,
    MedHybridTransformer,
    ResNet18Medical,
    build_medical_classifier,
    gradient_penalty_2d,
    vae_kl_free_bits,
    vae_per_sample_recon_error,
)


def test_w2_char_transformer_blocks_transfer_cleanly_to_med_hybrid(tmp_path: Path) -> None:
    # Given: A real W2 CharTransformer checkpoint saved with weights_only=True format.
    w2 = CharTransformer(vocab_size=76, d_model=256, n_heads=4, n_layers=4, dropout=0.1)
    w2_ckpt = tmp_path / "w2_model_best.pt"
    save_checkpoint(w2_ckpt, cpu_state_dict(w2), metadata={"model_hparams": w2.hparams})

    # When: Loading W2 blocks into MedHybridTransformer with freeze_blocks=True.
    med = MedHybridTransformer(
        num_classes=NUM_CLASSES, image_size=64, d_model=256, n_heads=4, n_layers=4
    )
    audit = med.load_w2_blocks(w2_ckpt, freeze_blocks=True)

    # Then: All Transformer block tensors match cleanly, text tensors are discarded, and blocks are frozen.
    assert audit["w2_checkpoint_found"] is True
    assert audit["transferred_block_tensors"] > 0
    assert audit["discarded_text_tensors"] >= 2  # token_embedding, position_embedding, fc
    assert audit["missing_keys"] == [] and audit["unexpected_keys"] == []
    assert all(not p.requires_grad for p in med.blocks.parameters())
    assert any(p.requires_grad for p in med.stem.parameters())

    # Forward pass and attention extraction match expected shapes.
    x = torch.randn(2, 3, 64, 64)
    logits, attns = med.forward_with_attention(x)
    assert logits.shape == (2, NUM_CLASSES)
    assert len(attns) == 4
    # 64/16 = 4 -> 4x4=16 patches + 1 CLS = 17 tokens
    assert attns[0].shape == (2, 4, 17, 17)


def test_med_hybrid_can_overfit_single_batch_and_reload_weights_only(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = MedHybridTransformer(
        num_classes=NUM_CLASSES, image_size=64, d_model=64, n_heads=4, n_layers=2
    )
    x = torch.randn(8, 3, 64, 64)
    y = torch.arange(8) % NUM_CLASSES
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    init_loss = float(loss_fn(model(x), y).item())
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    final_loss = float(loss_fn(model(x), y).item())
    assert final_loss < init_loss * 0.4

    # Checkpoint round-trip via weights_only=True
    ckpt = tmp_path / "med_test.pt"
    save_checkpoint(ckpt, cpu_state_dict(model), metadata={"model_hparams": model.hparams})
    sd, meta = load_checkpoint(ckpt, device="cpu")
    rebuilt = build_medical_classifier(meta["model_hparams"]).eval()
    rebuilt.load_state_dict(sd)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(x), rebuilt(x), atol=1e-5)


def test_resnet18_medical_forward_and_last_conv_layer() -> None:
    model = ResNet18Medical(num_classes=NUM_CLASSES, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)
    assert model.get_last_conv_layer() is not None


def test_w4_med_conv_vae_2d_forward_and_loss() -> None:
    vae = MedConvVAE2D(image_size=64, z_dim=32, base_ch=16)
    x = torch.rand(4, 3, 64, 64)
    recon, mu, log_var = vae(x, sample_posterior=True)
    assert recon.shape == x.shape
    assert float(recon.detach().min()) >= 0.0 and float(recon.detach().max()) <= 1.0
    rec = vae_per_sample_recon_error(recon, x)
    kl = vae_kl_free_bits(mu, log_var, free_bits=0.25)
    loss = rec.mean() + 0.002 * kl.mean()
    assert torch.isfinite(loss)
    loss.backward()


def test_w5_cond_wgan_gp_2d_second_order_gradient_penalty() -> None:
    device = torch.device("cpu")
    gen = CondGen2D(num_classes=NUM_CLASSES, noise_dim=32, class_emb_dim=8, base_ch=16).to(device)
    critic = CondCritic2D(num_classes=NUM_CLASSES, base_ch=16).to(device)

    real = torch.rand(4, 3, 64, 64, device=device) * 2.0 - 1.0
    labels = torch.tensor([0, 1, 4, 6], dtype=torch.long, device=device)
    z = torch.randn(4, 32, device=device)
    fake = gen(z, labels)
    assert fake.shape == real.shape

    gp = gradient_penalty_2d(critic, real, fake, labels, device)
    c_loss = critic(fake.detach(), labels).mean() - critic(real, labels).mean() + 10.0 * gp
    assert torch.isfinite(gp) and torch.isfinite(c_loss)
    c_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
