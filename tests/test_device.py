"""Pruebas unitarias de shared/device.py."""

from __future__ import annotations

import torch

from shared.device import (
    amp_context,
    clip_grad_norm_safe,
    dataloader_kwargs,
    get_hardware_profile,
    resolve_device,
    seed_everything,
)


def test_resolve_device_cpu_and_profile() -> None:
    device, desc = resolve_device("cpu")
    assert device.type == "cpu"
    assert "CPU" in desc
    prof = get_hardware_profile(device, "auto")
    assert prof.amp_enabled is False
    assert prof.use_grad_scaler is False


def test_amp_context_cpu_noop() -> None:
    device = torch.device("cpu")
    with amp_context(device):
        x = torch.randn(4, 4)
        y = x @ x.T
    assert y.dtype == torch.float32


def test_clip_grad_norm_safe_clips_and_detects_non_finite() -> None:
    param = torch.nn.Parameter(torch.ones(10))
    param.grad = torch.full((10,), 10.0)
    norm, is_finite = clip_grad_norm_safe([param], max_norm=1.0)
    assert is_finite is True
    assert norm > 1.0
    assert float(param.grad.norm().item()) <= 1.01

    param.grad = torch.full((10,), float("nan"))
    _, is_finite_nan = clip_grad_norm_safe([param], max_norm=1.0)
    assert is_finite_nan is False


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_dataloader_kwargs_cpu_safe() -> None:
    kw = dataloader_kwargs(torch.device("cpu"))
    assert kw["pin_memory"] is False
    assert kw["num_workers"] == 0
