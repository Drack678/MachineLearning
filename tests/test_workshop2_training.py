"""Pruebas unitarias de Fase 2 (Workshop 2 entrenamiento multi-token y checkpoint seguro)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shared.checkpoints import load_checkpoint, save_checkpoint
from shared.models import CharTransformer, build_warmup_cosine_scheduler
from Workshop_2.src.train import predict_logits, train_one_epoch


def test_w2_train_one_epoch_and_predict(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(42)
    blocks = torch.randint(1, 20, (16, 81), generator=g, dtype=torch.uint8)
    loader = DataLoader(TensorDataset(blocks), batch_size=4, shuffle=False)

    model = CharTransformer(vocab_size=20, d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = build_warmup_cosine_scheduler(opt, total_steps=12, warmup_ratio=0.2)
    crit = nn.CrossEntropyLoss()

    stats1 = train_one_epoch(
        model, loader, opt, sched, crit, torch.device("cpu"),
        amp_enabled=False, amp_dtype=torch.float32, scaler=None, grad_accum=2,
    )
    stats2 = train_one_epoch(
        model, loader, opt, sched, crit, torch.device("cpu"),
        amp_enabled=False, amp_dtype=torch.float32, scaler=None, grad_accum=2,
    )
    assert stats1["skipped_batches"] == 0.0
    assert stats2["train_bpc"] < stats1["train_bpc"]

    val_x = blocks[:, :-1]
    logits = predict_logits(model, val_x, torch.device("cpu"), batch_size=8)
    assert logits.shape == (16, 20)

    ckpt_path = tmp_path / "w2_best.pt"
    save_checkpoint(ckpt_path, model.state_dict(), metadata={"model_hparams": model.hparams})
    sd, meta = load_checkpoint(ckpt_path, device="cpu")
    restored = CharTransformer(**meta["model_hparams"])
    restored.load_state_dict(sd)
    assert torch.allclose(
        predict_logits(restored, val_x, torch.device("cpu")),
        logits,
        atol=1e-5,
    )
