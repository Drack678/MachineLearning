"""Persistencia segura de checkpoints (state_dict puro con weights_only=True + metadatos JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def metadata_path_for(checkpoint_path: Path | str) -> Path:
    """Devuelve la ruta del archivo JSON de metadatos asociado a un checkpoint .pt."""
    p = Path(checkpoint_path)
    return p.with_suffix(".meta.json")


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_checkpoint(
    path: Path | str,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Guarda tensores puros en ``path`` y metadatos en ``<stem>.meta.json``."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    pure_tensors = {
        k: v.detach().cpu().clone()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }
    if len(pure_tensors) != len(state_dict):
        non_tensors = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
        raise TypeError(
            f"save_checkpoint solo acepta tensores en state_dict; claves no tensor: {non_tensors}"
        )

    torch.save(pure_tensors, ckpt_path)
    meta_path = metadata_path_for(ckpt_path)
    meta_payload = _sanitize_json(metadata or {})
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
    return ckpt_path, meta_path


def load_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Carga un checkpoint seguro con ``weights_only=True`` y su archivo ``.meta.json``."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    meta_path = metadata_path_for(ckpt_path)
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return state_dict, metadata
