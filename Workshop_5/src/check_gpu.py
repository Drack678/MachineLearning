"""Diagnóstico de GPU AMD Radeon RX 9060 XT (ROCm / CUDA / DirectML) y autograd de 2do orden."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.device import DEVICE_CHOICES, is_directml, is_rocm_build, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifica detección GPU (AMD ROCm / CUDA / DirectML) y autograd de 2do orden."
    )
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    print(f"OS:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_version()}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"HIP/ROCm: {getattr(torch.version, 'hip', None)}")
    print(f"CUDA:     {torch.version.cuda}")

    device, desc = resolve_device(args.device)
    print(f"Selected: {device} | {desc}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        arch = getattr(props, "gcnArchName", "N/A")
        print(f"GPU:      {props.name}")
        print(f"VRAM:     {vram_gb:.2f} GB")
        if is_rocm_build():
            print(f"GCN Arch: {arch}")

    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = (a @ b).mean().item()
    print(f"[OK] Matmul en {device}: mean={c:.4f}")

    x = torch.randn(8, 16, device=device, requires_grad=True)
    w = torch.randn(16, 1, device=device, requires_grad=True)
    y = (x @ w).pow(2).sum()
    try:
        grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
        gp_like = ((grad_x.pow(2).sum(dim=1) + 1e-12).sqrt() - 1.0).pow(2).mean()
        gp_like.backward()
        print(f"[OK] Autograd de 2do orden (WGAN-GP): loss={gp_like.item():.4f}")
    except Exception as exc:
        print(f"[WARN] Falló autograd de 2do orden en {device}: {exc}")
        if is_directml(device):
            print("       DirectML tiene soporte limitado para gradient penalty; usa PyTorch + ROCm.")


if __name__ == "__main__":
    main()
