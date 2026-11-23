"""Pruebas unitarias de Fase 1 (Workshop 1 baselines en Pipeline + muestreo estratificado)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Workshop_1.src.baselines import (
    build_baseline_pipelines,
    predict_proba_aligned,
    stratified_sample_df,
)


def _make_mini_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    patterns = [
        ("the king said to th", 1),
        ("to be or not to be ", 2),
        ("What is your name? ", 3),
        ("O Romeo, Romeo! Wh ", 4),
    ]
    rows = []
    for i in range(160):
        txt, y = patterns[i % len(patterns)]
        prefix = " " * 61 + txt
        rows.append({"character_id": f"CHAR_{i % 6}", "x": prefix, "y_id": y})
    rows.append({"character_id": "CHAR_RARE", "x": " " * 61 + "rare singleton 1234", "y_id": 5})
    df = pd.DataFrame(rows)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def test_stratified_sample_preserves_rare_classes() -> None:
    df = _make_mini_df()
    sampled = stratified_sample_df(df, n_samples=60, seed=42)
    assert 5 in set(sampled["y_id"].tolist())
    assert set(sampled["y_id"].tolist()) == {1, 2, 3, 4, 5}


def test_pipelines_fit_predict_unknown_and_joblib_roundtrip(tmp_path: Path) -> None:
    df = _make_mini_df()
    X_train = df["x"].tolist()
    y_train = df["y_id"].to_numpy()
    X_test = [" " * 61 + "the king said to th", " " * 61 + "ZZZZZZZZZZ@@@@####!"]

    pipelines = build_baseline_pipelines(max_ngram_features=200, min_df=1, seed=42)
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        probs = predict_proba_aligned(pipe, X_test, vocab_size=10, chunk_size=1)
        assert probs.shape == (2, 10)
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

        out = tmp_path / f"{name}.joblib"
        joblib.dump(pipe, out)
        restored = joblib.load(out)
        probs2 = predict_proba_aligned(restored, X_test, vocab_size=10)
        assert np.allclose(probs, probs2, atol=1e-6)
