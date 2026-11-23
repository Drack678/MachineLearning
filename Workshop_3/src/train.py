"""Entrenamiento multi-semilla del agente Double+Dueling DQN (Workshop 3 — Fase 3)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.checkpoints import save_checkpoint
from shared.config import SEED, to_relative_path
from shared.device import DEVICE_CHOICES, cpu_state_dict, resolve_device, seed_everything
from Workshop_3.src.agent import DQNAgent
from Workshop_3.src.baselines import BASELINES, run_episodes
from Workshop_3.src.env import FIGURES_DIR, TextGenerationEnv, load_contexts

CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINTS_DIR / "agent_best.pt"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs" / "workshop3_v2"

N_TRAIN_CONTEXTS = 20_000
N_VAL_CONTEXTS = 150
EVAL_EVERY = 50
WARMUP_STEPS = 1_000
EPS_MIN = 0.05

CONFIGS: dict[str, dict[str, float]] = {
    "base":       {"lr": 5e-4, "gamma": 0.95, "eps_decay": 0.993, "tau": 0.005},
    "low_lr":     {"lr": 2e-4, "gamma": 0.95, "eps_decay": 0.993, "tau": 0.005},
    "high_gamma": {"lr": 5e-4, "gamma": 0.98, "eps_decay": 0.993, "tau": 0.005},
    "fast_decay": {"lr": 5e-4, "gamma": 0.95, "eps_decay": 0.988, "tau": 0.005},
}


def evaluate_agent(agent: DQNAgent, env: TextGenerationEnv, n: int) -> dict[str, Any]:
    return run_episodes(env, lambda obs, rng: agent.act(obs, 0.0), n)


def train_single_run(
    name: str,
    config: dict[str, float],
    episodes: int,
    train_env: TextGenerationEnv,
    val_env: TextGenerationEnv,
    device,
    *,
    seed: int,
    n_val: int,
    fast_verify: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed, deterministic=True)
    agent = DQNAgent(
        tabular_dim=train_env.tabular_dim,
        lm_hidden_dim=train_env.lm_hidden_dim,
        n_actions=train_env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        device=device,
    )
    writer = SummaryWriter(RUNS_DIR / f"{name}_s{seed}") if not fast_verify else None
    epsilon = 1.0
    steps = 0
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_quality": -np.inf, "val_return": -np.inf}
    start = time.time()

    for episode in range(1, episodes + 1):
        obs, _ = train_env.reset(seed=seed if episode == 1 else None)
        done = False
        total = 0.0
        losses: list[float] = []

        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            agent.buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total += reward
            steps += 1
            if steps >= WARMUP_STEPS and len(agent.buffer) >= agent.batch_size:
                losses.append(agent.update())

        epsilon = max(EPS_MIN, epsilon * config["eps_decay"])
        row: dict[str, Any] = {
            "config": name,
            "seed": seed,
            "episode": episode,
            "steps": steps,
            "return": total,
            "epsilon": epsilon,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "val_return": float("nan"),
            "val_mean_log_prob": float("nan"),
            "val_quality": float("nan"),
        }
        if writer is not None:
            writer.add_scalar("train/return", total, episode)
            writer.add_scalar("train/epsilon", epsilon, episode)
            if losses:
                writer.add_scalar("train/loss", row["loss"], episode)

        if episode % EVAL_EVERY == 0 or episode == episodes:
            ev = evaluate_agent(agent, val_env, n_val)
            row["val_return"] = ev["mean_return"]
            row["val_mean_log_prob"] = ev["mean_log_prob"]
            row["val_quality"] = ev["composite_quality"]
            if writer is not None:
                writer.add_scalar("val/return", ev["mean_return"], episode)
                writer.add_scalar("val/mean_log_prob", ev["mean_log_prob"], episode)
                writer.add_scalar("val/quality", ev["composite_quality"], episode)
            print(
                f"[{name}|s{seed}] ep {episode:03d} | val_ret {ev['mean_return']:.2f} "
                f"| val_lp {ev['mean_log_prob']:.3f} | quality {ev['composite_quality']:.2f} "
                f"| new {ev['new_word_rate']:.2%}"
            )
            if ev["composite_quality"] > best["val_quality"]:
                best = {
                    "config_name": name,
                    "seed": seed,
                    "episode": episode,
                    "steps": steps,
                    "val_return": ev["mean_return"],
                    "val_mean_log_prob": ev["mean_log_prob"],
                    "val_quality": ev["composite_quality"],
                    "q_hparams": agent.q.hparams,
                    "state_dict": cpu_state_dict(agent.q),
                }
        history.append(row)

    if writer is not None:
        writer.close()
    best["minutes"] = (time.time() - start) / 60.0
    return best, history


def plot_curves(history: list[dict[str, Any]], baselines_val: dict[str, dict[str, Any]]) -> None:
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, g in h.groupby("config", sort=False):
        by_ep = g.groupby("episode")["return"].mean()
        axes[0].plot(by_ep.index, by_ep.rolling(30, min_periods=1).mean(), label=name)
        v = g.dropna(subset=["val_quality"]).groupby("steps")[["val_quality", "val_mean_log_prob"]].mean()
        axes[1].plot(v.index, v["val_quality"], "o-", label=name)
        axes[2].plot(v.index, v["val_mean_log_prob"], "s-", label=name)

    for label, stats in baselines_val.items():
        axes[1].axhline(stats["composite_quality"], linestyle="--", alpha=0.6, label=label)
        axes[2].axhline(stats["mean_log_prob"], linestyle="--", alpha=0.6, label=label)

    axes[0].set(title="Training return (moving avg)", xlabel="episode")
    axes[1].set(title="Validation composite quality", xlabel="steps")
    axes[2].set(title="Validation fluency (mean log-prob)", xlabel="steps")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento Double+Dueling DQN (W3 Fase 3).")
    parser.add_argument("--config", choices=list(CONFIGS), default=None)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    parser.add_argument("--n-val", type=int, default=N_VAL_CONTEXTS)
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()

    if args.fast_verify:
        args.config = args.config or "base"
        args.episodes = min(args.episodes, 200)
        args.n_val = min(args.n_val, 60)

    seed_everything(args.seeds[0], deterministic=True)
    device, desc = resolve_device(args.device)
    print(f"Device: {device} | {desc}")

    train_env = TextGenerationEnv(load_contexts("train", N_TRAIN_CONTEXTS), device=device)
    val_env = TextGenerationEnv(
        load_contexts("val", args.n_val),
        device=device,
        lm=train_env.lm,
        vocab=train_env.vocab,
        dictionary=(train_env.words, train_env.prefixes),
    )

    baselines_val = {
        name: run_episodes(val_env, pol, args.n_val) for name, pol in BASELINES.items()
    }
    print(
        "Validation baselines:",
        {k: (round(v["mean_return"], 2), round(v["mean_log_prob"], 3)) for k, v in baselines_val.items()},
    )

    names = [args.config] if args.config else list(CONFIGS)
    all_history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    overall_best: dict[str, Any] | None = None

    for name in names:
        cfg = CONFIGS[name]
        seed_bests: list[dict[str, Any]] = []
        for s in args.seeds:
            best, hist = train_single_run(
                name, cfg, args.episodes, train_env, val_env, device,
                seed=s, n_val=args.n_val, fast_verify=args.fast_verify,
            )
            all_history.extend(hist)
            seed_bests.append(best)
            if not args.fast_verify:
                run_ckpt = CHECKPOINTS_DIR / f"agent_{name}_s{s}.pt"
                save_checkpoint(
                    run_ckpt,
                    best["state_dict"],
                    metadata={
                        "config_name": name,
                        "config": cfg,
                        "seed": s,
                        "episode": best["episode"],
                        "q_hparams": best["q_hparams"],
                        "val_return": best["val_return"],
                        "val_mean_log_prob": best["val_mean_log_prob"],
                        "val_quality": best["val_quality"],
                    },
                )
            if overall_best is None or best["val_quality"] > overall_best["val_quality"]:
                overall_best = best

        summary_rows.append(
            {
                "config": name,
                **cfg,
                "n_seeds": len(args.seeds),
                "val_return_mean": float(np.mean([b["val_return"] for b in seed_bests])),
                "val_return_std": float(np.std([b["val_return"] for b in seed_bests])),
                "val_log_prob_mean": float(np.mean([b["val_mean_log_prob"] for b in seed_bests])),
                "val_quality_mean": float(np.mean([b["val_quality"] for b in seed_bests])),
                "val_quality_std": float(np.std([b["val_quality"] for b in seed_bests])),
                "minutes_total": round(sum(b["minutes"] for b in seed_bests), 1),
            }
        )

    if overall_best is None:
        raise RuntimeError("No se produjo ningún agente válido.")

    if args.fast_verify:
        print(f"VAL_QUALITY={overall_best['val_quality']:.6f}")
        print(f"VAL_MEAN_LOG_PROB={overall_best['val_mean_log_prob']:.6f}")
        return

    save_checkpoint(
        CHECKPOINT,
        overall_best["state_dict"],
        metadata={
            "config_name": overall_best["config_name"],
            "config": CONFIGS[overall_best["config_name"]],
            "seed": overall_best["seed"],
            "episode": overall_best["episode"],
            "q_hparams": overall_best["q_hparams"],
            "val_return": overall_best["val_return"],
            "val_mean_log_prob": overall_best["val_mean_log_prob"],
            "val_quality": overall_best["val_quality"],
        },
    )
    print(
        f"\nPromovido a {to_relative_path(CHECKPOINT)}: {overall_best['config_name']} "
        f"(seed={overall_best['seed']}, quality={overall_best['val_quality']:.2f}, "
        f"val_lp={overall_best['val_mean_log_prob']:.3f})"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "hparam_search.csv", index=False)
    pd.DataFrame(all_history).to_csv(REPORTS_DIR / "training_history.csv", index=False)
    plot_curves(all_history, baselines_val)


if __name__ == "__main__":
    main()
