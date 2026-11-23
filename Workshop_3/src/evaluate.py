"""Evaluación en TEST del agente Double+Dueling DQN y baselines (Workshop 3 — Fase 3)."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import load_checkpoint
from shared.config import SEED
from shared.device import DEVICE_CHOICES, resolve_device, seed_everything
from Workshop_3.src.agent import DQNAgent
from Workshop_3.src.baselines import BASELINES, run_episodes
from Workshop_3.src.env import FIGURES_DIR, TextGenerationEnv, load_contexts
from Workshop_3.src.train import CHECKPOINT, REPORTS_DIR

N_TEST_CONTEXTS = 150


def load_agent_safe(path: Path, device) -> tuple[DQNAgent, dict[str, Any]]:
    state_dict, meta = load_checkpoint(path, device="cpu")
    qh = meta["q_hparams"]
    agent = DQNAgent(
        tabular_dim=qh["tabular_dim"],
        lm_hidden_dim=qh["lm_hidden_dim"],
        n_actions=qh["n_actions"],
        device=device,
    )
    agent.q.load_state_dict(state_dict)
    agent.q.eval()
    return agent, meta


def record_trajectory(env: TextGenerationEnv, agent: DQNAgent, index: int):
    obs, _ = env.reset(options={"index": index})
    steps: list[dict[str, Any]] = []
    done = False
    n_words = 0
    while not done:
        action = agent.act(obs, 0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cur_words = info["stats"]["new"] + info["stats"]["repeated"] + info["stats"]["invalid"]
        word_ended = cur_words > n_words
        n_words = cur_words
        steps.append(
            {
                "char": info["char"],
                "action": action,
                "word_reward": info["word_reward"],
                "word_ended": word_ended,
            }
        )
    return env.context, steps


def plot_trajectories(trajectories) -> None:
    fig, axes = plt.subplots(len(trajectories), 1, figsize=(14, 3.2 * len(trajectories)))
    if len(trajectories) == 1:
        axes = [axes]
    for ax, (context, steps) in zip(axes, trajectories, strict=True):
        ax.step(range(len(steps)), [s["action"] for s in steps], where="mid", color="steelblue")
        for t, s in enumerate(steps):
            if s["word_ended"]:
                color = "green" if s["word_reward"] > 0 else ("orange" if s["word_reward"] > -0.5 else "red")
                ax.scatter(t, 4.6, color=color, s=35, zorder=3)
        text = "".join(s["char"] for s in steps)
        ax.set_title(textwrap.fill(f"...{context[-40:]} || {text}", 140), fontsize=9, loc="left")
        ax.set_yticks(range(5))
        ax.set_ylim(-0.5, 5)
        ax.set_xlabel("step")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "trajectories.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación en TEST de Workshop 3 (Fase 3).")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--n-test", type=int, default=N_TEST_CONTEXTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    seed_everything(SEED)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    env = TextGenerationEnv(load_contexts("test", args.n_test), device=device)
    agent, meta = load_agent_safe(args.checkpoint, device)
    print(
        f"Loaded {args.checkpoint.name}: config={meta.get('config_name')}, "
        f"ep={meta.get('episode')}, val_quality={meta.get('val_quality', float('nan')):.2f}"
    )

    policies = dict(BASELINES)
    policies["Double+Dueling DQN (W3)"] = lambda obs, rng: agent.act(obs, 0.0)

    summary_rows: list[dict[str, Any]] = []
    sample_texts: dict[str, list[str]] = {}

    for name, policy in policies.items():
        per_seed = [run_episodes(env, policy, args.n_test, seed=s) for s in args.seeds]
        sample_texts[name] = per_seed[0]["texts"]
        metric_keys = [k for k in per_seed[0] if k != "texts"]
        row: dict[str, Any] = {"policy": name}
        for k in metric_keys:
            vals = [r[k] for r in per_seed]
            row[k] = float(np.mean(vals))
            if k in ("mean_return", "mean_log_prob", "composite_quality"):
                row[f"{k}_seed_std"] = float(np.std(vals))
        summary_rows.append(row)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summary_rows)
    table.to_csv(REPORTS_DIR / "test_comparison.csv", index=False)
    print("\nResultados en TEST (Workshop_3/reports/test_comparison.csv):")
    show_cols = [
        "policy", "mean_return", "mean_log_prob", "new_word_rate",
        "repeated_word_rate", "invalid_word_rate", "double_punct_rate", "composite_quality",
    ]
    print(table[show_cols].round(3).to_string(index=False))

    lines = []
    for i in range(min(5, args.n_test)):
        lines.append(f"Context: ...{env.contexts[i][-60:]!r}")
        lines.append(f"  greedy Transformer (W2): {sample_texts['greedy Transformer (W2)'][i]!r}")
        lines.append(f"  Double+Dueling DQN (W3): {sample_texts['Double+Dueling DQN (W3)'][i]!r}\n")
    (REPORTS_DIR / "trajectories.txt").write_text("\n".join(lines), encoding="utf-8")
    plot_trajectories([record_trajectory(env, agent, i) for i in range(3)])


if __name__ == "__main__":
    main()
