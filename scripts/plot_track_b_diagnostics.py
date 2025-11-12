from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_k_time_series(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, group in df.groupby("episode_mode"):
        ax.plot(group["global_step"], group["K"], label=name)
    ax.set_xlabel("Global step")
    ax.set_ylabel("K index")
    ax.set_title("K trajectories per episode mode")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "k_trajectories.png", dpi=200)
    plt.close(fig)


def plot_k_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, group in df.groupby("episode_mode"):
        ax.hist(group["K"], bins=40, alpha=0.4, label=name, density=True)
    ax.set_xlabel("K index")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of K by episode mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "k_distribution.png", dpi=200)
    plt.close(fig)


def plot_corridor_rate(df: pd.DataFrame, output_dir: Path) -> None:
    corridor = df["K"] >= 1.0
    grouped = df.assign(corridor=corridor).groupby("episode_mode")["corridor"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.sort_values().plot(kind="barh", ax=ax, color="tab:green")
    ax.set_xlabel("Corridor rate")
    ax.set_title("Corridor rate by episode mode")
    fig.tight_layout()
    fig.savefig(output_dir / "corridor_rates.png", dpi=200)
    plt.close(fig)


def plot_training_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(df["global_step"], df["actor_loss"], label="actor loss")
    axes[0].set_ylabel("Actor loss")
    axes[0].grid(alpha=0.2)

    axes[1].plot(df["global_step"], df["critic_loss"], color="tab:orange", label="critic loss")
    axes[1].set_ylabel("Critic loss")
    axes[1].grid(alpha=0.2)

    axes[2].plot(df["global_step"], df["alpha"], color="tab:green", label="alpha")
    axes[2].set_ylabel("Alpha")
    axes[2].set_xlabel("Global step")
    axes[2].grid(alpha=0.2)

    axes[0].set_title("SAC training metrics")
    for ax in axes:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_metrics.png", dpi=200)
    plt.close(fig)


def plot_metrics_by_mode(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for name, group in df.groupby("episode_mode"):
        axes[0].plot(group["global_step"], group["actor_loss"], label=name)
        axes[1].plot(group["global_step"], group["critic_loss"], label=name)
        axes[2].plot(group["global_step"], group["alpha"], label=name)

    axes[0].set_ylabel("Actor loss")
    axes[1].set_ylabel("Critic loss")
    axes[2].set_ylabel("Alpha")
    axes[2].set_xlabel("Global step")
    axes[0].set_title("Training metrics by episode mode")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_metrics_by_mode.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Track B diagnostics and training logs.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("logs/fre_track_b_diagnostics.csv"),
        help="CSV file with per-step diagnostics.",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("logs/fre_track_b_training.csv"),
        help="CSV file with SAC training metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/track_b"),
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diag_df = load_csv(args.diagnostics)
    training_df = load_csv(args.training)

    plot_k_time_series(diag_df, args.output_dir)
    plot_k_histogram(diag_df, args.output_dir)
    plot_corridor_rate(diag_df, args.output_dir)
    plot_training_metrics(training_df, args.output_dir)
    plot_metrics_by_mode(training_df, args.output_dir)


if __name__ == "__main__":
    main()
