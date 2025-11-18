"""Modern-era Historical K(t) prereg tests and quick correlations.

Implements:
- H1.1: Troughs below baseline − 1σ near prereg years
- H1.2: Positive trend 1950–1990 (p < 0.01)
- H1.3: Cross-correlation K(t) ~ wisdom_accuracy(t+10)
- H1.4: Higher pre-shock reciprocity predicts faster recovery
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
import json
import yaml


DATA_PATH = Path("logs/historical_k/k_t_series.csv")
CONFIG_PATH = Path("historical_k/k_config.yaml")


def load_series() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _nearest_decade(year: int, candidates: List[int]) -> int:
    return min(candidates, key=lambda y: abs(y - year))


def _prereg_events() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    return payload.get("preregistered_events", {})


def main() -> None:
    df_all = load_series()
    years = df_all["year"].tolist()

    # Baseline statistics
    mean_k = float(df_all["K"].mean())
    std_k = float(df_all["K"].std(ddof=0))

    # H1.1: Troughs < mean − 1σ
    events = _prereg_events()
    trough_years = events.get("troughs", [])
    trough_results: List[Tuple[int, float, bool]] = []
    for y in trough_years:
        y_near = _nearest_decade(int(y), years)
        k_val = float(df_all.loc[df_all["year"] == y_near, "K"].iloc[0])
        passed = k_val < (mean_k - std_k)
        trough_results.append((y_near, k_val, passed))

    # H1.2: Trend 1950–1990 positive and significant
    seg = df_all[(df_all["year"] >= 1950) & (df_all["year"] <= 1990)]
    lr = linregress(seg["year"].to_numpy(), seg["K"].to_numpy())
    trend_pass = lr.slope > 0 and lr.pvalue < 0.01

    # H1.3: Cross-correlation K(t) ~ wisdom_accuracy(t+10)
    w = df_all.set_index("year")["wisdom_accuracy"]
    k = df_all.set_index("year")["K"]
    # Align K(y) with W(y+10)
    common = sorted(set(k.index).intersection({y - 10 for y in w.index}))
    k_vec = k.loc[common]
    w_lead_vec = w.loc[[y + 10 for y in common]]
    if len(common) > 2 and (np.std(k_vec.to_numpy()) > 0) and (np.std(w_lead_vec.to_numpy()) > 0):
        r_kw, p_kw = pearsonr(k_vec.to_numpy(), w_lead_vec.to_numpy())
    else:
        r_kw, p_kw = (np.nan, np.nan)

    # H1.4: Pre-shock reciprocity predicts faster recovery
    shock_candidates = [1910, 1930, 1940]
    rec = df_all.set_index("year")["reciprocity"]

    def time_to_recover(shock_year: int) -> float:
        y0 = _nearest_decade(shock_year, years)
        y_pre = y0 - 10
        if y_pre not in rec.index or y0 not in k.index:
            return np.nan
        k_target = float(k.loc[y_pre])
        # Search up to 30 years after shock
        for d in (10, 20, 30):
            y_try = y0 + d
            if y_try in k.index and float(k.loc[y_try]) >= k_target:
                return float(d)
        return np.nan

    rows = []
    for sy in shock_candidates:
        y0 = _nearest_decade(sy, years)
        pre_rec = float(rec.get(y0 - 10, np.nan))
        tau = time_to_recover(sy)
        if not np.isnan(pre_rec) and not np.isnan(tau):
            rows.append((y0, pre_rec, tau))
    df_tau = pd.DataFrame(rows, columns=["shock_year", "pre_reciprocity", "tau_recovery_years"]).set_index("shock_year")
    if not df_tau.empty and len(df_tau) > 1:
        pre_arr = df_tau["pre_reciprocity"].to_numpy()
        neg_tau = (-df_tau["tau_recovery_years"]).to_numpy()
        if np.std(pre_arr) > 0 and np.std(neg_tau) > 0:
            r_tau, p_tau = pearsonr(pre_arr, neg_tau)
        else:
            r_tau, p_tau = (np.nan, np.nan)
    else:
        r_tau, p_tau = (np.nan, np.nan)

    # Report
    # Print summary
    print("H1.1 Troughs below mean-1σ:")
    for y, kv, ok in trough_results:
        print(f"  {y}: K={kv:.3f} → {'PASS' if ok else 'FAIL'}")
    print(f"H1.2 Trend 1950–1990: slope={lr.slope:.4f}, p={lr.pvalue:.4g} → {'PASS' if trend_pass else 'FAIL'}")
    print(f"H1.3 Corr[K(t), wisdom(t+10)]: r={r_kw:.3f}, p={p_kw:.4g}")
    if not df_tau.empty:
        print("H1.4 Pre-shock reciprocity vs. faster recovery:")
        for y, row in df_tau.iterrows():
            print(f"  {y}: preR={row['pre_reciprocity']:.3f}, tau={row['tau_recovery_years']:.1f}y")
        print(f"  Corr(preR, -tau): r={r_tau:.3f}, p={p_tau:.4g}")

    # Write JSON summary
    # Optional band-based trough check using per-year CI bands (if available)
    band_check = []
    summary_path = Path("logs/historical_k/k_t_summary.json")
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        low_arr = s.get("ci_low_yearly")
        if low_arr:
            # map year->CI_low per-year
            ci_low_by_year = {int(y): float(lo) for y, lo in zip(years, low_arr)}
            for y in trough_years:
                yn = _nearest_decade(int(y), years)
                kval = float(df_all.loc[df_all["year"] == yn, "K"].iloc[0])
                lo = ci_low_by_year.get(yn)
                if lo is not None:
                    band_check.append({"year": int(yn), "K": kval, "ci_low": float(lo), "pass": bool(kval < lo)})

    out = {
        "H1_1": {
            "troughs": [{"year": int(y), "K": float(kv), "pass": bool(ok)} for y, kv, ok in trough_results]
        },
        "H1_2": {"slope": float(lr.slope), "p": float(lr.pvalue), "pass": bool(trend_pass)},
        "H1_3": {"r": None if np.isnan(r_kw) else float(r_kw), "p": None if np.isnan(p_kw) else float(p_kw)},
        "H1_4": {
            "rows": [
                {
                    "shock_year": int(y),
                    "pre_reciprocity": float(row["pre_reciprocity"]),
                    "tau_recovery": float(row["tau_recovery_years"]),
                }
                for y, row in (df_tau.iterrows() if not df_tau.empty else [])
            ],
            "r": None if np.isnan(r_tau) else float(r_tau),
            "p": None if np.isnan(p_tau) else float(p_tau),
        },
        "H1_1_band": band_check,
    }
    out_path = Path("logs/historical_k/h1_checks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
