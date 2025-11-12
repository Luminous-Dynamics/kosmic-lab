from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_yaml_config
from core.kpass import KPassportWriter
from fre.simulate import simulate_phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fractal Reciprocity Engine Phase 1 experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to FRE configuration YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_bundle = load_yaml_config(args.config)
    schema_path = Path("schemas/k_passport.json")
    passport = KPassportWriter(schema_path=schema_path)

    te_cfg = config_bundle.payload.get("te", {})
    iit_cfg = config_bundle.payload.get("iit", {})
    estimators_base = {
        "phi": iit_cfg.get("primary_variant", "unknown"),
        "te": {
            "estimator": te_cfg.get("estimator", "unknown"),
            "k": int(te_cfg.get("k", 0)),
            "lag": int(te_cfg.get("lag", 0)),
        },
    }

    runs = simulate_phase1(config_bundle.payload)
    output_dir = Path("logs/fre_phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        params = run["parameters"]
        metrics = run["metrics"]
        estimators = dict(estimators_base)
        record = passport.build_record(
            experiment="fre_phase1",
            params=params,
            estimators=estimators,
            metrics=metrics,
            config=config_bundle,
            seed=run["seed"],
        )
        passport.write(record, output_dir)

    print(f"[FRE] Completed {len(runs)} runs. Passports saved to {output_dir}")


if __name__ == "__main__":
    main()
