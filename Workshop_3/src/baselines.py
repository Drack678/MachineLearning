"""Políticas base y evaluador de episodios para Workshop 3 (Fase 3)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Workshop_3.src.env import K, PER_CAND_FEATURES, TextGenerationEnv


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(K))


def greedy_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    return 0


def sampling_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    probs = obs[0 : PER_CAND_FEATURES * K : PER_CAND_FEATURES].astype(np.float64)
    total = float(probs.sum())
    if total <= 0.0:
        return 0
    return int(rng.choice(K, p=probs / total))


BASELINES: dict[str, Callable[[np.ndarray, np.random.Generator], int]] = {
    "random": random_policy,
    "greedy Transformer (W2)": greedy_policy,
    "sampling Transformer (W2)": sampling_policy,
}


def run_episodes(
    env: TextGenerationEnv,
    policy: Callable[[np.ndarray, np.random.Generator], int],
    n_episodes: int,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    log_probs: list[float] = []
    all_word_lens: list[int] = []
    stats = {"new": 0, "repeated": 0, "invalid": 0, "double_punct": 0}
    total_steps = 0
    texts: list[str] = []

    for i in range(n_episodes):
        obs, _ = env.reset(options={"index": i})
        done = False
        total = 0.0
        info: dict[str, Any] = {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, rng))
            done = terminated or truncated
            total += reward
            log_probs.append(float(info["log_prob"]))
            total_steps += 1
        returns.append(total)
        texts.append(env.generated)
        for key in stats:
            stats[key] += int(info["stats"][key])
        all_word_lens.extend(info.get("word_lengths", []))

    words = max(1, stats["new"] + stats["repeated"] + stats["invalid"])
    mean_return = float(np.mean(returns))
    mean_lp = float(np.mean(log_probs))
    return {
        "mean_return": mean_return,
        "std_return": float(np.std(returns)),
        "new_word_rate": stats["new"] / words,
        "repeated_word_rate": stats["repeated"] / words,
        "invalid_word_rate": stats["invalid"] / words,
        "double_punct_rate": stats["double_punct"] / max(1, total_steps),
        "words_per_episode": (stats["new"] + stats["repeated"] + stats["invalid"]) / n_episodes,
        "mean_word_len": float(np.mean(all_word_lens)) if all_word_lens else 0.0,
        "mean_log_prob": mean_lp,
        "composite_quality": mean_return + 5.0 * mean_lp,
        "texts": texts,
    }
