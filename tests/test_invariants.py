"""Pruebas de invariantes de datos de Workshop 1 (splits.json y vocab.json intocables)."""

from __future__ import annotations

from shared.config import (
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL_CHARACTERS,
    EXPECTED_VOCAB_SIZE,
    REPO_ROOT,
    to_relative_path,
    verify_data_invariants,
)
from shared.data import reconstruct_character_segments


def test_splits_and_vocab_invariants_intact() -> None:
    report = verify_data_invariants()
    assert report["overlaps"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    assert report["total_characters"] == EXPECTED_TOTAL_CHARACTERS
    assert report["vocab_size"] == EXPECTED_VOCAB_SIZE
    assert report["split_counts"] == EXPECTED_SPLIT_COUNTS


def test_to_relative_path_portable() -> None:
    p = REPO_ROOT / "Workshop_2" / "checkpoints" / "model_best.pt"
    assert to_relative_path(p) == "Workshop_2/checkpoints/model_best.pt"


def test_reconstruct_character_segments_never_crosses_speakers(synthetic_blocks_df) -> None:
    segments = reconstruct_character_segments(synthetic_blocks_df)
    assert len(segments) == 2
    assert "cat" in segments[0] and "cab" not in segments[0]
    assert "cab" in segments[1] and "cat" not in segments[1]
