"""Fachada compatible de selección de dispositivo delegando a `shared.device`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.device import (
    DEVICE_CHOICES,
    amp_context,
    clip_grad_norm_safe,
    cpu_state_dict,
    dataloader_kwargs,
    get_amp_config,
    is_directml,
    is_rocm_build,
    resolve_device,
    seed_everything,
)


def print_device_info(device, description: str) -> None:
    print(f"Device: {device} | {description}")


__all__ = [
    "DEVICE_CHOICES",
    "amp_context",
    "clip_grad_norm_safe",
    "cpu_state_dict",
    "dataloader_kwargs",
    "get_amp_config",
    "is_directml",
    "is_rocm_build",
    "print_device_info",
    "resolve_device",
    "seed_everything",
]
