"""Entorno Gymnasium de generación controlable de texto (Workshop 3 — Fase 3)."""

from __future__ import annotations

import re
import string
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint
from shared.config import SEED, W2_BEST_CHECKPOINT, WINDOW_SIZE
from shared.data import load_split_df, read_vocab
from shared.device import get_amp_config, resolve_device, seed_everything
from shared.models import CharTransformer

WORDS_FILE = ROOT / "data" / "words_clean.txt"
FIGURES_DIR = ROOT / "reports" / "figures"

WORD_CHARS = set(string.ascii_letters + "'")
VALID_SINGLE_CHAR_WORDS = {"a", "i", "o"}
K = 5
EPISODE_LEN = 100
PER_CAND_FEATURES = 6
GLOBAL_FEATURES = 4
TABULAR_OBS_DIM = PER_CAND_FEATURES * K + GLOBAL_FEATURES


def set_seeds(seed: int = SEED) -> None:
    seed_everything(seed, deterministic=True)


def get_device(preferred: str = "auto") -> torch.device:
    device, _ = resolve_device(preferred)
    return device


def is_valid_dictionary_word(token: str) -> bool:
    w = token.lower()
    if not w:
        return False
    if w.startswith("'") or w.endswith("'"):
        return False
    if not any(ch.isalpha() for ch in w):
        return False
    if len(w) == 1:
        return w in VALID_SINGLE_CHAR_WORDS
    return True


def extract_trailing_word(context: str) -> str:
    match = re.search(r"[A-Za-z']+$", context)
    return match.group(0) if match else ""


def load_contexts(split: str, n: int, seed: int = SEED) -> list[str]:
    df = load_split_df(split, columns=["x"])  # type: ignore[arg-type]
    idx = np.random.default_rng(seed).choice(len(df), size=min(n, len(df)), replace=False)
    return df["x"].iloc[idx].tolist()


def load_dictionary(words_file: Path = WORDS_FILE) -> tuple[set[str], set[str]]:
    if not words_file.exists():
        df = load_split_df("train", columns=["x"])
        raw_words: set[str] = set()
        for window in df["x"].iloc[::40]:
            tokens = re.findall(r"[A-Za-z']+", window)[1:-1]
            for tok in tokens:
                if is_valid_dictionary_word(tok):
                    raw_words.add(tok.lower())
        words_file.parent.mkdir(parents=True, exist_ok=True)
        words_file.write_text("\n".join(sorted(raw_words)) + "\n", encoding="utf-8")

    words = {w for w in words_file.read_text(encoding="utf-8").split() if is_valid_dictionary_word(w)}
    prefixes = {w[:i] for w in words for i in range(1, len(w) + 1)}
    return words, prefixes


def load_language_model(
    device: torch.device, checkpoint_path: Path = W2_BEST_CHECKPOINT
) -> tuple[CharTransformer, dict[str, Any]]:
    state_dict, meta = load_checkpoint(checkpoint_path, device="cpu")
    model = CharTransformer(**meta["model_hparams"])
    model.load_state_dict(state_dict)
    vocab = read_vocab()
    return model.to(device).eval(), vocab


class TextGenerationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        contexts: list[str],
        device: torch.device | None = None,
        render_mode: str | None = None,
        *,
        fluency_weight: float = 0.25,
        repeat_penalty: float = -0.25,
        punct_penalty: float = 0.30,
        episode_len: int = EPISODE_LEN,
        lm: CharTransformer | None = None,
        vocab: dict[str, Any] | None = None,
        dictionary: tuple[set[str], set[str]] | None = None,
        checkpoint_path: Path = W2_BEST_CHECKPOINT,
    ) -> None:
        super().__init__()
        self.contexts = contexts
        self.device = device or get_device()
        self.render_mode = render_mode
        self.fluency_weight = float(fluency_weight)
        self.repeat_penalty = float(repeat_penalty)
        self.punct_penalty = float(punct_penalty)
        self.episode_len = int(episode_len)
        self.amp_enabled, self.amp_dtype, _ = get_amp_config(self.device)

        if lm is not None and vocab is not None:
            self.lm = lm.to(self.device).eval()
            self.vocab = vocab
        else:
            self.lm, self.vocab = load_language_model(self.device, checkpoint_path)

        if dictionary is not None:
            self.words, self.prefixes = dictionary
        else:
            self.words, self.prefixes = load_dictionary()

        self.lm_hidden_dim = int(self.lm.hparams["d_model"])
        self.tabular_dim = TABULAR_OBS_DIM
        self.action_space = spaces.Discrete(K)
        obs_dim = self.tabular_dim + self.lm_hidden_dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.context = ""
        self.generated = ""
        self.word = ""
        self.used: set[str] = set()
        self.steps = 0
        self.prev_was_punct = False
        self.stats: dict[str, int] = {}
        self.word_lengths: list[int] = []
        self.last_reward = 0.0
        self.cand_chars: list[str] = []
        self.cand_probs: np.ndarray = np.zeros(K, dtype=np.float32)
        self.lm_hidden: np.ndarray = np.zeros(self.lm_hidden_dim, dtype=np.float32)

    @torch.inference_mode()
    def _update_candidates(self) -> None:
        text = (self.context + self.generated)[-WINDOW_SIZE:].rjust(WINDOW_SIZE)
        stoi = self.vocab["stoi"]
        unk_idx = int(self.vocab["unk_index"])
        ids = torch.tensor([[stoi.get(c, unk_idx) for c in text]], device=self.device, dtype=torch.long)

        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled and self.device.type == "cuda"
        ):
            hidden = self.lm.features(ids)
            logits = self.lm.fc(hidden)[0].float()

        probs = torch.softmax(logits, dim=-1)
        probs[unk_idx] = 0.0
        probs_sum = float(probs.sum().item())
        if probs_sum > 0.0:
            probs = probs / probs_sum

        values, indices = probs.topk(K)
        self.cand_chars = [self.vocab["itos"][i] for i in indices.tolist()]
        self.cand_probs = values.cpu().numpy().astype(np.float32)

        h_vec = hidden[0].float()
        h_vec = h_vec / (h_vec.norm(p=2).clamp_min(1e-6))
        self.lm_hidden = h_vec.cpu().numpy().astype(np.float32)

    def _obs(self) -> np.ndarray:
        features: list[float] = []
        word = self.word.lower()
        for char, p in zip(self.cand_chars, self.cand_probs, strict=True):
            is_space = char == " "
            is_word_char = char in WORD_CHARS
            is_punct = (not is_space) and (not is_word_char)
            ends_word = not is_word_char

            log_p_clipped = float(np.clip(np.log(float(p) + 1e-12), -4.0, 0.0))
            log_p_norm = (log_p_clipped + 4.0) / 4.0

            if ends_word:
                valid = (word == "" and not (is_punct and self.prev_was_punct)) or (
                    word in self.words
                )
                repeated = word != "" and word in self.used
            else:
                valid = (word + char.lower()) in self.prefixes
                repeated = False

            features.extend(
                [
                    float(p),
                    log_p_norm,
                    float(is_space),
                    float(is_punct),
                    float(valid),
                    float(repeated),
                ]
            )

        features.extend(
            [
                min(len(word) / 10.0, 1.0),
                self.steps / float(self.episode_len),
                float(self.prev_was_punct),
                min(len(self.used) / 20.0, 1.0),
            ]
        )
        tab = np.asarray(features, dtype=np.float32)
        return np.concatenate([tab, self.lm_hidden], axis=0)

    def _score_word(self) -> float:
        word = self.word.lower()
        self.word_lengths.append(len(word))
        if word not in self.words:
            self.stats["invalid"] += 1
            return -1.0
        if word in self.used:
            self.stats["repeated"] += 1
            return self.repeat_penalty
        self.used.add(word)
        self.stats["new"] += 1
        return 1.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "index" in options:
            index = int(options["index"])
        else:
            index = int(self.np_random.integers(len(self.contexts)))
        self.context = self.contexts[index]
        self.generated = ""
        self.word = extract_trailing_word(self.context)
        last_ctx_char = self.context[-1] if self.context else " "
        self.prev_was_punct = (last_ctx_char != " ") and (last_ctx_char not in WORD_CHARS)
        self.used = set()
        self.steps = 0
        self.stats = {"new": 0, "repeated": 0, "invalid": 0, "double_punct": 0}
        self.word_lengths = []
        self.last_reward = 0.0
        self._update_candidates()
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        char = self.cand_chars[int(action)]
        log_prob = float(np.log(float(self.cand_probs[int(action)]) + 1e-12))
        is_word_char = char in WORD_CHARS
        is_punct = (char != " ") and (not is_word_char)

        word_reward = 0.0
        if is_word_char:
            self.word += char
        elif self.word:
            word_reward = self._score_word()
            self.word = ""

        punct_pen = 0.0
        if is_punct and self.prev_was_punct:
            self.stats["double_punct"] += 1
            punct_pen = self.punct_penalty

        fluency_term = self.fluency_weight * float(np.clip(log_prob, -4.0, 0.0))
        reward = float(word_reward + fluency_term - punct_pen)

        self.prev_was_punct = is_punct
        self.generated += char
        self.steps += 1
        self.last_reward = reward

        terminated = False
        truncated = self.steps >= self.episode_len
        if not truncated:
            self._update_candidates()
        if self.render_mode == "human":
            self.render()

        info = {
            "char": char,
            "log_prob": log_prob,
            "word_reward": word_reward,
            "fluency_term": fluency_term,
            "punct_penalty": punct_pen,
            "stats": dict(self.stats),
            "word_lengths": list(self.word_lengths),
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> None:
        print(f"...{self.context[-30:]!r} + {self.generated!r} (r={self.last_reward:+.2f})")


def main() -> None:
    set_seeds()
    env = TextGenerationEnv(load_contexts("val", 50))
    env.action_space.seed(SEED)
    returns = []
    for ep in range(30):
        obs, _ = env.reset(seed=SEED + ep)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            total += reward
        returns.append(total)
    print(f"30 random episodes | mean return={np.mean(returns):.2f} ± {np.std(returns):.2f}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(returns, bins=12)
    ax.set_title("Random policy return (Fase 3)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_policy.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
