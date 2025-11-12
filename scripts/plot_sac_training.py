"""Generate SAC training curves for reward and entropy."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "artifacts" / "sac_run_01" / "evaluations.npz"
LOG_PATH = ROOT / "sac_train.log"
OUTPUT_DIR = ROOT / "figs" / "trackBC_success_curves"


def plot_evaluation_rewards() -> None:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(EVAL_PATH)

    data = np.load(EVAL_PATH)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_eval, episodes)
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(timesteps, mean_rewards, label="mean reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="tab:blue",
        alpha=0.2,
        label="±1σ",
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Evaluation reward")
    ax.set_title("SAC Evaluation Reward Curve")
    ax.legend()
    ax.grid(alpha=0.2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reward_curve.png", dpi=200)
    plt.close(fig)


def parse_ent_coef_series() -> tuple[np.ndarray, np.ndarray]:
    if not LOG_PATH.exists():
        raise FileNotFoundError(LOG_PATH)

    timesteps: list[float] = []
    entropies: list[float] = []
    last_timestep: float | None = None

    timestep_pattern = re.compile(r"total_timesteps\s*\|\s*([0-9.eE+-]+)")
    ent_pattern = re.compile(r"ent_coef\s*\|\s*([0-9.eE+-]+)")

    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            match_ts = timestep_pattern.search(line)
            if match_ts:
                try:
                    last_timestep = float(match_ts.group(1))
                except ValueError:
                    continue
                continue

            match_ent = ent_pattern.search(line)
            if match_ent and last_timestep is not None:
                try:
                    ent_value = float(match_ent.group(1))
                except ValueError:
                    continue
                timesteps.append(last_timestep)
                entropies.append(ent_value)

    if not timesteps:
        raise RuntimeError("No entropy coefficients parsed from sac_train.log")

    return np.asarray(timesteps), np.asarray(entropies)


def plot_entropy_coefficients() -> None:
    timesteps, entropies = parse_ent_coef_series()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(timesteps, entropies, color="tab:orange")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Entropy coefficient")
    ax.set_title("SAC Entropy Coefficient")
    ax.grid(alpha=0.2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entropy_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    plot_evaluation_rewards()
    plot_entropy_coefficients()


if __name__ == "__main__":
    main()
