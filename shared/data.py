"""Carga vectorizada de splits, cache binario uint8 (weights_only=True) y bloques contiguos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from shared.config import (
    CACHE_DIR,
    SPLITS_FILE,
    UNK_INDEX,
    VOCAB_FILE,
    W1_PROCESSED_DIR,
    WINDOW_SIZE,
)

SplitName = Literal["train", "val", "test"]


def read_vocab(vocab_path: Path = VOCAB_FILE) -> dict[str, Any]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"No se encontró el vocabulario en {vocab_path}")
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def read_splits(splits_path: Path = SPLITS_FILE) -> dict[str, Any]:
    if not splits_path.exists():
        raise FileNotFoundError(f"No se encontró splits.json en {splits_path}")
    return json.loads(splits_path.read_text(encoding="utf-8"))


def build_ascii_lookup(stoi: dict[str, int]) -> np.ndarray:
    table = np.full(256, UNK_INDEX, dtype=np.uint8)
    for char, idx in stoi.items():
        if len(char) == 1 and ord(char) < 256:
            table[ord(char)] = idx
    return table


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(c, UNK_INDEX) for c in text]


def encode_windows_vectorized(
    series: pd.Series, stoi: dict[str, int], window: int = WINDOW_SIZE
) -> torch.Tensor:
    table = build_ascii_lookup(stoi)
    raw = np.frombuffer("".join(series.tolist()).encode("latin-1"), dtype=np.uint8)
    arr = table[raw].reshape(-1, window).copy()
    return torch.from_numpy(arr)


def load_split_df(split: SplitName, columns: list[str] | None = None) -> pd.DataFrame:
    path = W1_PROCESSED_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta `python -m src.dataset --prepare` en Workshop_1."
        )
    return pd.read_parquet(path, columns=columns)


def load_encoded_split(
    split: SplitName,
    vocab: dict[str, Any] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = cache_dir / f"{split}_encoded_u8.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["x"], data["y"]

    vocab = vocab or read_vocab()
    df = load_split_df(split, columns=["x", "y_id"])
    x = encode_windows_vectorized(df["x"], vocab["stoi"], window=WINDOW_SIZE)
    y = torch.from_numpy(df["y_id"].to_numpy(dtype=np.uint8, copy=True))

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, cache_path)
    return x, y


def reconstruct_character_segments(df: pd.DataFrame) -> list[str]:
    segments: list[str] = []
    for _, group in df.groupby("character_id", sort=False):
        xs = group["x"].tolist()
        ys = group["y"].tolist()
        if not xs:
            continue
        buf = [xs[0], ys[0]]
        prev_x = xs[0]
        prev_y = ys[0]
        for cur_x, cur_y in zip(xs[1:], ys[1:], strict=True):
            if cur_x == prev_x[1:] + prev_y:
                buf.append(cur_y)
            else:
                segments.append("".join(buf))
                buf = [cur_x, cur_y]
            prev_x = cur_x
            prev_y = cur_y
        segments.append("".join(buf))
    return segments


def build_contiguous_blocks(
    split: SplitName,
    vocab: dict[str, Any] | None = None,
    block_size: int = WINDOW_SIZE + 1,
    stride: int = 40,
    cache_dir: Path = CACHE_DIR,
) -> torch.Tensor:
    if block_size < 2:
        raise ValueError("block_size debe ser >= 2")
    if stride < 1:
        raise ValueError("stride debe ser >= 1")

    cache_path = cache_dir / f"{split}_blocks_b{block_size}_s{stride}_u8.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["blocks"]

    vocab = vocab or read_vocab()
    table = build_ascii_lookup(vocab["stoi"])
    df = load_split_df(split, columns=["character_id", "x", "y"])
    segments = reconstruct_character_segments(df)

    blocks_list: list[np.ndarray] = []
    for seg in segments:
        if len(seg) < block_size:
            continue
        encoded = table[np.frombuffer(seg.encode("latin-1"), dtype=np.uint8)]
        n_len = len(encoded)
        starts = list(range(0, n_len - block_size + 1, stride))
        last_start = n_len - block_size
        if starts[-1] != last_start:
            starts.append(last_start)
        for s in starts:
            blocks_list.append(encoded[s : s + block_size])

    blocks_np = np.stack(blocks_list, axis=0) if blocks_list else np.empty((0, block_size), dtype=np.uint8)
    blocks = torch.from_numpy(blocks_np)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"blocks": blocks}, cache_path)
    return blocks
