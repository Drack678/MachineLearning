"""Configuración global, rutas portables e invariantes de datos de Workshop 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEED: int = 42
WINDOW_SIZE: int = 80
TARGET_SIZE: int = 1
UNK_TOKEN: str = "<UNK>"
UNK_INDEX: int = 0

EXPECTED_VOCAB_SIZE: int = 76
EXPECTED_TOTAL_CHARACTERS: int = 1129
EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "train": 790,
    "val": 169,
    "test": 170,
}

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
WORKSHOP_DIRS: dict[int, Path] = {
    i: REPO_ROOT / f"Workshop_{i}" for i in range(1, 7)
}
W1_DIR: Path = WORKSHOP_DIRS[1]
W2_DIR: Path = WORKSHOP_DIRS[2]
W3_DIR: Path = WORKSHOP_DIRS[3]
W4_DIR: Path = WORKSHOP_DIRS[4]
W5_DIR: Path = WORKSHOP_DIRS[5]

W1_DATA_DIR: Path = W1_DIR / "data"
W1_PROCESSED_DIR: Path = W1_DATA_DIR / "processed"
SPLITS_FILE: Path = W1_DATA_DIR / "splits.json"
VOCAB_FILE: Path = W1_DATA_DIR / "vocab.json"
CACHE_DIR: Path = REPO_ROOT / "shared" / ".cache"

W2_BEST_CHECKPOINT: Path = W2_DIR / "checkpoints" / "model_best.pt"


def to_relative_path(path: Path | str) -> str:
    """Convierte una ruta absoluta dentro del repo en una ruta POSIX relativa portable."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def verify_data_invariants(
    splits_path: Path = SPLITS_FILE,
    vocab_path: Path = VOCAB_FILE,
) -> dict[str, Any]:
    """Verifica que splits.json y vocab.json mantengan intactos los invariantes de Workshop 1."""
    if not splits_path.exists():
        raise FileNotFoundError(f"No existe splits.json en {splits_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"No existe vocab.json en {vocab_path}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    char_ids = splits["character_ids"]
    sets = {name: set(char_ids[name]) for name in ("train", "val", "test")}
    overlaps = {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Fuga de datos detectada entre splits: {overlaps}")

    for name, expected_n in EXPECTED_SPLIT_COUNTS.items():
        actual_n = len(char_ids[name])
        if actual_n != expected_n:
            raise AssertionError(
                f"Split {name!r} alterado: {actual_n} personajes vs {expected_n} esperados"
            )

    total_chars = sum(len(v) for v in char_ids.values())
    if total_chars != EXPECTED_TOTAL_CHARACTERS:
        raise AssertionError(
            f"Total de personajes alterado: {total_chars} vs {EXPECTED_TOTAL_CHARACTERS}"
        )

    if vocab["size"] != EXPECTED_VOCAB_SIZE or len(vocab["itos"]) != EXPECTED_VOCAB_SIZE:
        raise AssertionError(
            f"Vocabulario alterado: size={vocab['size']} vs {EXPECTED_VOCAB_SIZE}"
        )
    if vocab["itos"][UNK_INDEX] != UNK_TOKEN or vocab["stoi"].get(UNK_TOKEN) != UNK_INDEX:
        raise AssertionError("El índice 0 del vocabulario debe corresponder a '<UNK>'")

    return {
        "overlaps": overlaps,
        "total_characters": total_chars,
        "vocab_size": vocab["size"],
        "split_counts": {k: len(v) for k, v in char_ids.items()},
    }
