"""Gestión unificada de dispositivos (AMD ROCm, NVIDIA CUDA, DirectML, CPU), AMP y reproducibilidad."""

from __future__ import annotations

import contextlib
import math
import os
import platform
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

DEVICE_CHOICES: tuple[str, ...] = ("auto", "rocm", "cuda", "dml", "cpu")
PROFILE_CHOICES: tuple[str, ...] = ("auto", "rocm-rx9060", "low-vram", "cpu")


@dataclass(frozen=True)
class HardwareProfile:
    """Parámetros recomendados de entrenamiento según la GPU disponible."""

    name: str
    batch_size: int
    grad_accum: int
    amp_enabled: bool
    amp_dtype: torch.dtype
    use_grad_scaler: bool
    use_foreach: bool | None


def _try_import_directml():
    try:
        import torch_directml  # type: ignore[import-not-found]

        return torch_directml
    except ImportError:
        return None


def is_directml(device: torch.device) -> bool:
    return device.type == "privateuseone"


def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _cuda_or_rocm_label() -> str:
    gpu_name = torch.cuda.get_device_name(0)
    if is_rocm_build():
        return f"AMD ROCm/HIP ({gpu_name}, hip={torch.version.hip})"
    return f"NVIDIA CUDA ({gpu_name}, cuda={torch.version.cuda})"


def resolve_device(preferred: str = "auto") -> tuple[torch.device, str]:
    """Resuelve el dispositivo de cómputo priorizando ROCm/CUDA -> DirectML -> CPU."""
    choice = (preferred or "auto").lower()
    if choice not in DEVICE_CHOICES:
        raise ValueError(
            f"Dispositivo no válido: {preferred!r}. Opciones: {', '.join(DEVICE_CHOICES)}"
        )

    if choice == "cpu":
        return torch.device("cpu"), "CPU"

    if choice in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No se detectó GPU compatible con ROCm/CUDA en PyTorch. "
                "Usa `--device auto` o instala PyTorch con soporte ROCm/CUDA."
            )
        if choice == "rocm" and not is_rocm_build():
            raise RuntimeError(
                "Se solicitó `--device rocm`, pero PyTorch no tiene soporte HIP/ROCm."
            )
        return torch.device("cuda"), _cuda_or_rocm_label()

    if choice == "dml":
        dml = _try_import_directml()
        if dml is None or not dml.is_available():
            raise RuntimeError("`torch-directml` no está disponible.")
        idx = dml.default_device()
        return dml.device(idx), f"DirectML ({dml.device_name(idx)})"

    if torch.cuda.is_available():
        return torch.device("cuda"), _cuda_or_rocm_label()

    dml = _try_import_directml()
    if dml is not None and dml.is_available():
        idx = dml.default_device()
        return dml.device(idx), f"DirectML ({dml.device_name(idx)})"

    return torch.device("cpu"), "CPU (sin aceleración GPU detectada)"


def get_amp_config(device: torch.device) -> tuple[bool, torch.dtype, bool]:
    """Retorna (amp_enabled, dtype, use_grad_scaler) seguro para el dispositivo."""
    if device.type != "cuda":
        return False, torch.float32, False

    if is_rocm_build():
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: True)())
        if bf16_ok:
            return True, torch.bfloat16, False
        return True, torch.float16, True

    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_ok:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def get_hardware_profile(device: torch.device, profile: str = "auto") -> HardwareProfile:
    """Devuelve el perfil de memoria y precisión para el dispositivo dado."""
    choice = (profile or "auto").lower()
    if choice not in PROFILE_CHOICES:
        raise ValueError(f"Perfil no válido: {profile!r}. Opciones: {', '.join(PROFILE_CHOICES)}")

    amp_enabled, amp_dtype, use_scaler = get_amp_config(device)
    use_foreach = False if is_rocm_build() else None

    if choice == "cpu" or device.type != "cuda":
        return HardwareProfile(
            name="cpu",
            batch_size=64,
            grad_accum=1,
            amp_enabled=False,
            amp_dtype=torch.float32,
            use_grad_scaler=False,
            use_foreach=None,
        )

    if choice == "low-vram":
        return HardwareProfile(
            name="low-vram",
            batch_size=64,
            grad_accum=4,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_scaler,
            use_foreach=use_foreach,
        )

    if choice == "rocm-rx9060":
        return HardwareProfile(
            name="rocm-rx9060",
            batch_size=256,
            grad_accum=1,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_scaler,
            use_foreach=False,
        )

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb <= 6.0:
        return HardwareProfile(
            name="low-vram",
            batch_size=64,
            grad_accum=4,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_scaler,
            use_foreach=use_foreach,
        )
    return HardwareProfile(
        name="rocm-rx9060" if is_rocm_build() else "high-vram",
        batch_size=256,
        grad_accum=1,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_scaler,
        use_foreach=use_foreach,
    )


@contextlib.contextmanager
def amp_context(
    device: torch.device,
    enabled: bool | None = None,
    dtype: torch.dtype | None = None,
) -> Iterator[None]:
    """Context manager portable para mixed precision (seguro en ROCm, CUDA, DirectML y CPU)."""
    default_enabled, default_dtype, _ = get_amp_config(device)
    use_amp = default_enabled if enabled is None else (enabled and device.type == "cuda")
    target_dtype = default_dtype if dtype is None else dtype
    if not use_amp:
        yield
        return
    with torch.amp.autocast(device_type="cuda", dtype=target_dtype, enabled=True):
        yield


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Fija semillas de Python, NumPy, PyTorch y CUDA/ROCm para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copia el state_dict en CPU para checkpoints portables."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def clip_grad_norm_safe(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float = 1.0,
) -> tuple[float, bool]:
    """Clipea gradientes de forma segura en ROCm (evita glitches de _foreach_norm en gfx1200)."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0, True

    if is_rocm_build():
        total_sq = 0.0
        for p in params:
            val = p.grad.detach().float().pow(2).sum().item()
            if not math.isfinite(val):
                return float("nan"), False
            total_sq += val
        if not math.isfinite(total_sq):
            return float("nan"), False
        grad_norm = math.sqrt(total_sq)
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-6)
            for p in params:
                p.grad.mul_(scale)
        return grad_norm, True

    grad_norm_t = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    grad_norm = float(grad_norm_t.item()) if isinstance(grad_norm_t, torch.Tensor) else float(grad_norm_t)
    return grad_norm, math.isfinite(grad_norm)


def dataloader_kwargs(device: torch.device, num_workers: int | None = None) -> dict[str, object]:
    """Argumentos seguros de DataLoader para ROCm, CUDA, DirectML y CPU."""
    is_win = platform.system() == "Windows"
    use_pin = device.type == "cuda" and not (is_rocm_build() and is_win)
    if num_workers is None:
        workers = 0 if is_win or device.type != "cuda" else min(4, os.cpu_count() or 1)
    else:
        workers = num_workers
    kwargs: dict[str, object] = {"pin_memory": use_pin, "num_workers": workers}
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs
