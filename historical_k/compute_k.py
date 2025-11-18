from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import matplotlib.pyplot as plt

from core.config import load_yaml_config
from core.kpass import KPassportWriter
from historical_k.etl import build_harmony_frame, compute_k_series, load_feature_series, normalize_series
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Historical K(t) from configured proxies."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Historical K(t) configuration YAML.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        help=(
            "Override normalization strategy (e.g., minmax_by_century, "
            "zscore_by_century, zscore_rolling_3, minmax_rolling_3)."
        ),
    )
    parser.add_argument(
        "--per-year-bands",
        action="store_true",
        help="Force-enable per-year bootstrap CI bands (resample across features).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Override bootstrap sample count for CI calculations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed for bootstrap.",
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
    resampled_means = [float(rng.choice(data, size=data.size, replace=True).mean()) for _ in range(samples)]
    alpha = 1 - float(cfg.get("ci", 0.95))
    lower = float(np.percentile(resampled_means, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lower, upper

def _plot_series(
    results_df,
    ci_low,
    ci_high,
    path: Path,
    *,
    prereg_events: Dict[str, Any] | None = None,
    reference_line: float | None = None,
) -> None:
    """Plot K(t) with uncertainty bands.

    ci_low/ci_high can be either scalars or arrays aligned to results_df["year"].
    """
    years = results_df["year"].to_numpy()
    k_vals = results_df["K"].to_numpy()
    # Broadcast scalar CI into arrays if needed
    if isinstance(ci_low, (int, float)):
        ci_low_arr = pd.Series([ci_low] * len(years))
    else:
        ci_low_arr = pd.Series(ci_low)
    if isinstance(ci_high, (int, float)):
        ci_high_arr = pd.Series([ci_high] * len(years))
    else:
        ci_high_arr = pd.Series(ci_high)

    plt.figure(figsize=(7, 4))
    plt.plot(years, k_vals, marker="o", label="K(t)")
    plt.fill_between(years, ci_low_arr, ci_high_arr, color="orange", alpha=0.2, label="Bootstrap CI")
    if reference_line is not None:
        plt.axhline(reference_line, color="red", linestyle="--", label="Reference")
    plt.xlabel("Year")
    plt.ylabel("K-index")
    plt.title("Historical K(t)")
    plt.legend()
    # Mark preregistered events if provided
    if prereg_events:
        xs = results_df["year"].to_numpy()
        for label in ("troughs", "peaks"):
            for y in (prereg_events.get(label) or []):
                if xs.min() <= y <= xs.max():
                    color = "blue" if label == "troughs" else "green"
                    plt.axvline(y, color=color, linestyle=":", alpha=0.6)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _plot_harmonies(results_df, path: Path) -> None:
    """Plot per-harmony normalized series for diagnostics."""
    years = results_df["year"].to_numpy()
    harmony_cols = [
        "resonant_coherence",
        "interconnection",
        "reciprocity",
        "play_entropy",
        "wisdom_accuracy",
        "flourishing",
    ]
    plt.figure(figsize=(8, 5))
    for col in harmony_cols:
        if col in results_df.columns:
            plt.plot(years, results_df[col].to_numpy(), label=col)
    plt.xlabel("Year")
    plt.ylabel("Normalized value")
    plt.title("Harmony diagnostics")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _build_feature_mats(proxies: Dict[str, Any], years: list[int], normalization: str) -> Dict[str, pd.DataFrame]:
    """Load all proxy series into per-harmony DataFrames (columns=features).

    This enables bootstrap resampling across features within each harmony to
    produce per-year uncertainty bands.
    """
    mats: Dict[str, pd.DataFrame] = {}
    for harmony, features in proxies.items():
        cols = {}
        for feat in features:
            raw = load_feature_series(feat, years)
            cols[feat] = normalize_series(raw, years, normalization)
        mats[harmony] = pd.DataFrame(cols)
    return mats


def _bootstrap_bands_per_year(
    feature_mats: Dict[str, pd.DataFrame],
    samples: int,
    ci: float,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute per-year bootstrap CI bands by resampling features within harmonies.

    For each bootstrap sample, for each harmony, resample its feature columns
    with replacement and average to a harmony series; then average across
    harmonies to get a K(t) series. Percentiles across samples yield bands.
    """
    if samples <= 0:
        # No bootstrap → return NaNs; caller can fallback to scalar CI
        first = next(iter(feature_mats.values()))
        n = len(first.index)
        return pd.Series([float("nan")] * n, index=first.index), pd.Series([float("nan")] * n, index=first.index)

    rng = np.random.default_rng(seed)
    years_index = next(iter(feature_mats.values())).index
    harmony_names = list(feature_mats.keys())

    all_k = []  # list of arrays length = n_years
    for _ in range(samples):
        harmony_series = []
        for h in harmony_names:
            mat = feature_mats[h]
            if mat.shape[1] == 0:
                # Should not happen; guard just in case
                harmony_series.append(pd.Series(0.0, index=years_index))
                continue
            # Resample feature columns with replacement
            cols = mat.columns.to_list()
            resampled_cols = [cols[i] for i in rng.integers(0, len(cols), size=len(cols))]
            resampled = mat[resampled_cols]
            harmony_series.append(resampled.mean(axis=1))
        # Average harmonies to get K(t)
        k_t = pd.concat(harmony_series, axis=1).mean(axis=1)
        all_k.append(k_t.to_numpy())

    all_k_arr = np.vstack(all_k)  # shape: [samples, n_years]
    alpha = 1 - ci
    low = np.percentile(all_k_arr, 100 * (alpha / 2), axis=0)
    high = np.percentile(all_k_arr, 100 * (1 - alpha / 2), axis=0)
    return pd.Series(low, index=years_index), pd.Series(high, index=years_index)


def main() -> None:
    args = parse_args()
    config_bundle = load_yaml_config(args.config)
    validate_config(config_bundle.payload)

    years = _years_from_config(config_bundle.payload)
    windows_cfg = config_bundle.payload.get("windows", {})
    normalization = args.normalization or windows_cfg.get("normalization", "none")
    normalization_overrides = windows_cfg.get("normalization_overrides", {})
    agg_cfg = (config_bundle.payload.get("aggregation", {}) or {})
    feature_aggregation = agg_cfg.get("feature", "mean")
    feature_agg_over = agg_cfg.get("feature_overrides", {})
    harmony_frame = build_harmony_frame(
        config_bundle.payload.get("proxies", {}),
        years,
        normalization,
        normalization_overrides,
        feature_aggregation,
        feature_agg_over,
    )
    weights_cfg = (config_bundle.payload.get("weighting", {}) or {}).get("harmonies", {})
    k_series = compute_k_series(harmony_frame, weights=weights_cfg or None)
    results = harmony_frame.copy()
    results["K"] = k_series
    results = results.reset_index(names="year")
    output_dir = Path("logs/historical_k")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "k_t_series.csv"
    results.to_csv(output_path, index=False)

    uncertainty_cfg = dict(config_bundle.payload.get("uncertainty", {}))
    # Apply CLI overrides to uncertainty config if provided
    if args.bootstrap_samples is not None:
        uncertainty_cfg["bootstrap_samples"] = int(args.bootstrap_samples)
    if args.seed is not None:
        uncertainty_cfg["seed"] = int(args.seed)
    if args.per_year_bands:
        uncertainty_cfg["per_year"] = True
    # Scalar CI around mean for backwards-compatibility
    ci_low_scalar, ci_high_scalar = _bootstrap_mean_ci(k_series, uncertainty_cfg)

    # Optional per-year bands via feature-level bootstrap
    per_year = bool(uncertainty_cfg.get("per_year", False))
    seed = int(uncertainty_cfg.get("seed", 0))
    ci = float(uncertainty_cfg.get("ci", 0.95))
    samples = int(uncertainty_cfg.get("bootstrap_samples", 0))
    if per_year and samples > 0:
        feature_mats = _build_feature_mats(config_bundle.payload.get("proxies", {}), years, normalization)
        ci_low_year, ci_high_year = _bootstrap_bands_per_year(feature_mats, samples, ci, seed)
        ci_low_plot, ci_high_plot = ci_low_year, ci_high_year
    else:
        ci_low_year = None
        ci_high_year = None
        ci_low_plot, ci_high_plot = ci_low_scalar, ci_high_scalar

    summary = {
        "mean_K": float(k_series.mean()),
        "ci_low": float(ci_low_scalar),
        "ci_high": float(ci_high_scalar),
        "years": [years[0], years[-1]],
        "bootstrap_samples": samples,
        "per_year": per_year,
        "normalization": normalization,
        "feature_aggregation": feature_aggregation,
        "weights": weights_cfg,
    }
    if ci_low_year is not None and ci_high_year is not None:
        summary["ci_low_yearly"] = [float(x) for x in ci_low_year.to_numpy()]
        summary["ci_high_yearly"] = [float(x) for x in ci_high_year.to_numpy()]
    summary_path = output_dir / "k_t_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Choose reference line based on normalization if not specified in config
    ref_line = 0.0 if str(normalization).startswith("zscore") else (0.5 if str(normalization).startswith("minmax") else 1.0)

    _plot_series(
        results,
        ci_low_plot,
        ci_high_plot,
        output_dir / "k_t_plot.png",
        prereg_events=config_bundle.payload.get("preregistered_events", {}),
        reference_line=ref_line,
    )

    _plot_harmonies(results, output_dir / "k_t_harmonies.png")

    schema_path = Path("schemas/k_passport.json")
    passport = KPassportWriter(schema_path=schema_path)
    record = passport.build_record(
        experiment="historical_k_v1",
        params={"years": [years[0], years[-1]]},
        estimators={"phi": "historical_proxy", "te": {"estimator": "none", "k": 0, "lag": 0}},
        metrics={"K": float(k_series.mean())},
        config=config_bundle,
        ci={"mean_low": ci_low_scalar, "mean_high": ci_high_scalar},
    )
    passport.write(record, output_dir)
    print(f"[Historical K] Series saved to {output_path}; summary -> {summary_path}")


if __name__ == "__main__":
    main()
