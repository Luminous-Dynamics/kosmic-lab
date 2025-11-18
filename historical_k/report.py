from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path("logs/historical_k")


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_json(OUT_DIR / "k_t_summary.json") or {}
    h1 = _read_json(OUT_DIR / "h1_checks.json") or {}
    comp = _read_json(OUT_DIR / "compare_summary.json") or {}

    lines = []
    lines.append("# Historical K(t) Report")
    lines.append("")
    # Summary
    if summary:
        lines.append("## Summary")
        lines.append(f"- Years: {summary.get('years')}")
        lines.append(f"- Normalization: {summary.get('normalization')}")
        lines.append(f"- mean(K): {summary.get('mean_K'):.4f}")
        lines.append(f"- 95% CI (mean): [{summary.get('ci_low'):.4f}, {summary.get('ci_high'):.4f}]")
        lines.append("")
    # Plots
    lines.append("## Plots")
    if (OUT_DIR / "k_t_plot.png").exists():
        lines.append("- K(t) with CI: k_t_plot.png")
    if (OUT_DIR / "k_t_harmonies.png").exists():
        lines.append("- Harmony diagnostics: k_t_harmonies.png")
    if (OUT_DIR / "k_compare.png").exists():
        lines.append("- Normalization comparison: k_compare.png")
    if (OUT_DIR / "contrib_rolling.png").exists():
        lines.append("- Harmony rolling correlations: contrib_rolling.png")
    lines.append("")
    # H1 checks
    if h1:
        lines.append("## H1 Checks")
        h11 = h1.get("H1_1", {}).get("troughs", [])
        lines.append("- H1.1 Troughs: " + ", ".join([f"{t['year']} -> {'PASS' if t['pass'] else 'FAIL'}" for t in h11]))
        h12 = h1.get("H1_2", {})
        lines.append(f"- H1.2 Trend 1950–1990: slope={h12.get('slope'):.4f}, p={h12.get('p'):.4g} -> {'PASS' if h12.get('pass') else 'FAIL'}")
        h13 = h1.get("H1_3", {})
        r13 = h13.get("r")
        p13 = h13.get("p")
        lines.append(f"- H1.3 K vs wisdom(t+10): r={r13 if r13 is not None else 'NA'}, p={p13 if p13 is not None else 'NA'}")
        lines.append("")
    # Comparison
    if comp:
        lines.append("## Normalization Comparison")
        for k, v in comp.items():
            lines.append(f"- {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, slope_1950_1990={v['slope_1950_1990']:.4f}")
        lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT_DIR / "report.md")


if __name__ == "__main__":
    main()

