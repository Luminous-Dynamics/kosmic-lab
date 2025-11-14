from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

import matplotlib.pyplot as plt

from core.config import load_yaml_config
from core.kpass import KPassportWriter
from historical_k.etl import build_harmony_frame, compute_k_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Historical K(t) from configured proxies.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Historical K(t) configuration YAML.",
    )
    return parser.parse_args()


def validate_config(payload: Dict[str, Any]) -> None:
    required_top_level = {"windows", "proxies"}
    missing = required_top_level.difference(payload)
    if missing:
        raise ValueError(f"Historical config missing keys: {sorted(missing)}")

    proxies = payload["proxies"]
    if not isinstance(proxies, dict) or not proxies:
        raise ValueError("Historical config requires non-empty 'proxies' mapping.")
    for harmony, features in proxies.items():
        if not isinstance(features, Iterable) or not list(features):
            raise ValueError(f"Harmony '{harmony}' defined without proxy features.")


def _years_from_config(payload: Dict[str, Any]) -> List[int]:
    window_cfg = payload.get("windows", {})
    size = window_cfg.get("size", "decade")
    if size != "decade":
        raise ValueError(f"Unsupported window size: {size}")
    # Default span derived from preregistered events if available.
    events = payload.get("preregistered_events", {})
    candidate_years = []
    for vals in events.values():
        candidate_years.extend(vals)
    if candidate_years:
        start = int(min(candidate_years) // 10 * 10)
        end = int(max(candidate_years) // 10 * 10)
    else:
        start, end = 1800, 2020
    return list(range(start, end + 10, 10))


def _bootstrap_mean_ci(k_series, cfg: Dict[str, Any]) -> tuple[float, float]:
    samples = int(cfg.get("bootstrap_samples", 0))
    if samples <= 0:
        mean_value = float(k_series.mean())
        return mean_value, mean_value

    rng = np.random.default_rng(cfg.get("seed", 0))
    data = k_series.to_numpy()
    resampled_means = [
        float(rng.choice(data, size=data.size, replace=True).mean()) for _ in range(samples)
    ]
    alpha = 1 - float(cfg.get("ci", 0.95))
    lower = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lower, upper


def _plot_series(results_df, summary: Dict[str, Any], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(results_df["year"], results_df["K"], marker="o", label="K(t)")
    plt.fill_between(
        results_df["year"],
        summary["ci_low"],
        summary["ci_high"],
        color="orange",
        alpha=0.2,
        label="Bootstrap CI",
    )
    plt.axhline(1.0, color="red", linestyle="--", label="Corridor threshold")
    plt.xlabel("Year")
    plt.ylabel("K-index")
    plt.title("Historical K(t)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    config_bundle = load_yaml_config(args.config)
    validate_config(config_bundle.payload)

    years = _years_from_config(config_bundle.payload)
    harmony_frame = build_harmony_frame(config_bundle.payload.get("proxies", {}), years)
    k_series = compute_k_series(harmony_frame)
    results = harmony_frame.copy()
    results["K"] = k_series
    results = results.reset_index(names="year")
    output_dir = Path("logs/historical_k")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "k_t_series.csv"
    results.to_csv(output_path, index=False)

    uncertainty_cfg = config_bundle.payload.get("uncertainty", {})
    ci_low, ci_high = _bootstrap_mean_ci(k_series, uncertainty_cfg)

    summary = {
        "mean_K": float(k_series.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "years": [years[0], years[-1]],
        "bootstrap_samples": int(uncertainty_cfg.get("bootstrap_samples", 0)),
    }
    summary_path = output_dir / "k_t_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _plot_series(results, summary, output_dir / "k_t_plot.png")

    schema_path = Path("schemas/k_passport.json")
    passport = KPassportWriter(schema_path=schema_path)
    record = passport.build_record(
        experiment="historical_k_v1",
        params={"years": [years[0], years[-1]]},
        estimators={"phi": "historical_proxy", "te": {"estimator": "none", "k": 0, "lag": 0}},
        metrics={"K": float(k_series.mean())},
        config=config_bundle,
        ci={"mean_low": ci_low, "mean_high": ci_high},
    )
    passport.write(record, output_dir)
    print(f"[Historical K] Series saved to {output_path}; summary -> {summary_path}")


if __name__ == "__main__":
    main()
