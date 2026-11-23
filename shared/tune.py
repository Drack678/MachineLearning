"""Runner acotado de Optuna y configuraciones `autoresearch` para Fase 6 (Workshops 2–5)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import SEED, verify_data_invariants

AUTORESEARCH_SPECS: dict[str, dict[str, str]] = {
    "w2": {
        "goal": "Minimizar val_bpc (bits-per-character) del Transformer multi-token en Workshop 2",
        "scope": "shared/models.py, Workshop_2/src/train.py",
        "metric": "val_bpc (lower is better)",
        "verify": "python -m Workshop_2.src.train --fast-verify --epochs 5 --seed 42 --device rocm | grep 'VAL_BPC='",
        "guard": "pytest tests/test_invariants.py tests/test_shared_models.py tests/test_workshop2_training.py",
        "max_iterations": "20",
    },
    "w3": {
        "goal": "Maximizar val_quality = val_return + 5.0 * val_mean_log_prob sin reward hacking en Workshop 3",
        "scope": "Workshop_3/src/agent.py, Workshop_3/src/train.py",
        "metric": "val_quality (higher is better)",
        "verify": "python -m Workshop_3.src.train --fast-verify --episodes 200 --seeds 42 --device rocm | grep 'VAL_QUALITY='",
        "guard": "pytest tests/test_invariants.py tests/test_workshop3_env.py",
        "max_iterations": "15",
    },
    "w4": {
        "goal": "Maximizar val_char_acc del VAE secuencial manteniendo val_kl >= 15.0 (cero posterior collapse)",
        "scope": "Workshop_4/src/autoencoder.py, Workshop_4/src/train_ae.py",
        "metric": "val_vae_score (higher is better)",
        "verify": "python -m Workshop_4.src.train_ae --fast-verify --epochs 10 --beta 1.0 --device rocm | grep 'VAL_VAE_SCORE='",
        "guard": "pytest tests/test_invariants.py tests/test_workshop4_ae.py",
        "max_iterations": "15",
    },
    "w5": {
        "goal": "Minimizar bigram_distribution_tv en validación manteniendo unique_sequence_ratio >= 0.95",
        "scope": "Workshop_5/src/gan.py, Workshop_5/src/train_wgan.py",
        "metric": "val_bigram_tv (lower is better)",
        "verify": "python -m Workshop_5.src.train_wgan --fast-verify --epochs 5 --device rocm | grep 'VAL_BIGRAM_TV='",
        "guard": "pytest tests/test_invariants.py tests/test_workshop5_gan.py",
        "max_iterations": "15",
    },
}


def format_autoresearch_block(workshop: str) -> str:
    spec = AUTORESEARCH_SPECS[workshop]
    return (
        f"### {workshop.upper()} (ciclo acotado: {spec['max_iterations']} iteraciones)\n"
        f"Goal:   {spec['goal']}\n"
        f"Scope:  {spec['scope']}\n"
        f"Metric: {spec['metric']}\n"
        f"Verify: {spec['verify']}\n"
        f"Guard:  {spec['guard']}\n"
    )


def _run_and_parse_metric(cmd: list[str], pattern: str) -> float:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    output = proc.stdout + "\n" + proc.stderr
    match = re.search(pattern, output)
    if proc.returncode != 0 or match is None:
        raise RuntimeError(f"Fallo al ejecutar {' '.join(cmd)}:\n{output[-1000:]}")
    return float(match.group(1))


def run_optuna_study(workshop: str, n_trials: int, device: str, timeout: int) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna no está instalado. Instálalo con `pip install optuna` para usar `--workshop`."
        ) from exc

    verify_data_invariants()
    py = sys.executable

    if workshop == "w2":
        direction = "minimize"

        def objective(trial: Any) -> float:
            lr = trial.suggest_float("lr", 3e-4, 2.5e-3, log=True)
            wd = trial.suggest_float("weight_decay", 1e-2, 1.5e-1, log=True)
            do = trial.suggest_float("dropout", 0.05, 0.25)
            cmd = [
                py, "-m", "Workshop_2.src.train", "--fast-verify",
                "--epochs", "5", "--lr", f"{lr:.6g}", "--weight-decay", f"{wd:.6g}",
                "--dropout", f"{do:.4g}", "--device", device, "--seed", str(SEED),
            ]
            return _run_and_parse_metric(cmd, r"VAL_BPC=([0-9.eE+-]+)")

    elif workshop == "w4":
        direction = "maximize"

        def objective(trial: Any) -> float:
            lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
            fb = trial.suggest_float("free_bits", 0.10, 0.50)
            cycles = trial.suggest_int("kl_cycles", 2, 6)
            cmd = [
                py, "-m", "Workshop_4.src.train_ae", "--fast-verify",
                "--epochs", "10", "--beta", "1.0", "--lr", f"{lr:.6g}",
                "--free-bits", f"{fb:.4g}", "--kl-cycles", str(cycles),
                "--device", device, "--seed", str(SEED),
            ]
            return _run_and_parse_metric(cmd, r"VAL_VAE_SCORE=([0-9.eE+-]+)")

    elif workshop == "w5":
        direction = "minimize"

        def objective(trial: Any) -> float:
            lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
            lgp = trial.suggest_float("lambda_gp", 5.0, 20.0)
            csteps = trial.suggest_int("critic_steps", 3, 7)
            cmd = [
                py, "-m", "Workshop_5.src.train_wgan", "--fast-verify",
                "--epochs", "5", "--learning-rate", f"{lr:.6g}",
                "--lambda-gp", f"{lgp:.4g}", "--critic-steps", str(csteps),
                "--device", device, "--seed", str(SEED),
            ]
            return _run_and_parse_metric(cmd, r"VAL_BIGRAM_TV=([0-9.eE+-]+)")

    else:
        direction = "maximize"

        def objective(trial: Any) -> float:
            cfg = trial.suggest_categorical("config", ["base", "low_lr", "high_gamma", "fast_decay"])
            cmd = [
                py, "-m", "Workshop_3.src.train", "--fast-verify",
                "--config", str(cfg), "--episodes", "200",
                "--device", device, "--seeds", str(SEED),
            ]
            return _run_and_parse_metric(cmd, r"VAL_QUALITY=([0-9.eE+-]+)")

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return {"best_value": study.best_value, "best_params": study.best_params}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna y Autoresearch configs (Fase 6).")
    parser.add_argument("--print-autoresearch", action="store_true")
    parser.add_argument("--workshop", choices=("w2", "w3", "w4", "w5"), default=None)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.print_autoresearch or args.workshop is None:
        for w in ("w2", "w3", "w4", "w5"):
            print(format_autoresearch_block(w))
        return

    res = run_optuna_study(args.workshop, args.n_trials, args.device, args.timeout)
    print(f"[{args.workshop}] Best value={res['best_value']:.6f} | params={res['best_params']}")


if __name__ == "__main__":
    main()
