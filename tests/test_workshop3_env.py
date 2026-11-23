"""Pruebas unitarias de Fase 3 (MDP corregido, diccionario limpio y Double+Dueling DQN)."""

from __future__ import annotations

import numpy as np
import torch

from shared.models import CharTransformer
from Workshop_3.src.agent import DQNAgent
from Workshop_3.src.env import (
    TABULAR_OBS_DIM,
    TextGenerationEnv,
    extract_trailing_word,
    is_valid_dictionary_word,
)


def _mock_vocab() -> dict:
    itos = ["<UNK>", " ", "e", "s", "t", "a", "h", "-", "?", "'"]
    return {"size": len(itos), "itos": itos, "stoi": {c: i for i, c in enumerate(itos)}, "unk_index": 0}


def test_dictionary_filter_rejects_degenerate_tokens() -> None:
    assert is_valid_dictionary_word("the") is True
    assert is_valid_dictionary_word("a") is True
    assert is_valid_dictionary_word("i") is True
    assert is_valid_dictionary_word("'s") is False
    assert is_valid_dictionary_word("'t") is False
    assert is_valid_dictionary_word("'ll") is False
    assert is_valid_dictionary_word("g") is False
    assert is_valid_dictionary_word("'") is False


def test_extract_trailing_word_fixes_boundary_bug() -> None:
    assert extract_trailing_word("Welcome, dear cousin") == "cousin"
    assert extract_trailing_word("a kind of hard consci") == "consci"
    assert extract_trailing_word("ended with space ") == ""


def test_env_boundary_truncation_and_double_punct_penalty() -> None:
    vocab = _mock_vocab()
    lm = CharTransformer(vocab_size=vocab["size"], d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    words = {"the", "hat", "set", "a", "cousin"}
    prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}

    env = TextGenerationEnv(
        contexts=[" " * 60 + "Welcome, dear cousin"],
        device=torch.device("cpu"),
        episode_len=5,
        lm=lm,
        vocab=vocab,
        dictionary=(words, prefixes),
        punct_penalty=0.30,
    )

    obs, _ = env.reset(options={"index": 0})
    assert obs.shape == (TABULAR_OBS_DIM + 16,)
    assert env.word == "cousin"

    env.cand_chars = ["-", "?", " ", "e", "t"]
    env.cand_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    _, _, term1, trunc1, info1 = env.step(0)
    assert info1["word_reward"] == 1.0
    assert info1["punct_penalty"] == 0.0
    assert term1 is False and trunc1 is False

    env.cand_chars = ["?", "-", " ", "e", "t"]
    env.cand_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    _, _, _, _, info2 = env.step(0)
    assert info2["punct_penalty"] == 0.30
    assert info2["stats"]["double_punct"] == 1

    terminated, truncated = False, False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is True


def test_double_dueling_dqn_bootstraps_and_updates() -> None:
    agent = DQNAgent(
        tabular_dim=TABULAR_OBS_DIM,
        lm_hidden_dim=16,
        n_actions=5,
        batch_size=8,
        device="cpu",
    )
    rng = np.random.default_rng(42)
    for _ in range(16):
        s = rng.standard_normal(TABULAR_OBS_DIM + 16).astype(np.float32)
        ns = rng.standard_normal(TABULAR_OBS_DIM + 16).astype(np.float32)
        agent.buffer.add(s, int(rng.integers(5)), 0.5, ns, terminated=False)

    loss = agent.update()
    assert np.isfinite(loss)
    act = agent.act(rng.standard_normal(TABULAR_OBS_DIM + 16).astype(np.float32), epsilon=0.0)
    assert 0 <= act < 5
