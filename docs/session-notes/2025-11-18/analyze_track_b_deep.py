#!/usr/bin/env python3
"""
Deep Analysis of Track B Logs

Properly extract and analyze the Track B experimental data.
"""

import numpy as np
from pathlib import Path
from collections import Counter


def compute_h2_diversity(actions):
    """H2: Action distribution entropy."""
    if len(actions) < 5:
        return 0.5
    # Discretize continuous actions
    if actions.dtype in [np.float32, np.float64]:
        # Bin continuous actions
        discretized = np.digitize(actions.flatten(), bins=np.linspace(-1, 1, 10))
    else:
        discretized = actions.flatten()

    counts = Counter(discretized)
    if len(counts) <= 1:
        return 0.0
    probs = np.array([c / len(discretized) for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    return float(h / np.log(len(counts)))


def compute_h1_from_states(states):
    """H1: Resonant coherence from state correlations."""
    if len(states) < 10 or states.ndim == 1:
        return 0.5

    n_dims = states.shape[1]
    if n_dims < 2:
        return 0.5

    correlations = []
    for i in range(min(n_dims, 8)):
        for j in range(i+1, min(n_dims, 8)):
            if np.std(states[:, i]) > 1e-10 and np.std(states[:, j]) > 1e-10:
                r = np.corrcoef(states[:, i], states[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))

    return float(np.mean(correlations)) if correlations else 0.5


def analyze_episode(filepath):
    """Analyze a single episode's NPZ file."""
    data = np.load(filepath, allow_pickle=True)

    result = {
        'file': filepath.name,
        'mode': 'open_loop' if 'open_loop' in filepath.name else
                'train' if 'train' in filepath.name else 'eval'
    }

    # Get K values (these are Simple K!)
    if 'K_values' in data:
        k_vals = np.array(data['K_values'])
        if k_vals.ndim > 1:
            k_vals = k_vals.flatten()
        result['simple_k_mean'] = float(np.mean(k_vals))
        result['simple_k_final'] = float(k_vals[-1]) if len(k_vals) > 0 else 0
        result['simple_k_max'] = float(np.max(k_vals))

    # Get states for H1 calculation
    if 'states' in data:
        states = np.array(data['states'])
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        result['h1'] = compute_h1_from_states(states)
        result['n_steps'] = len(states)
        result['state_dim'] = states.shape[1] if states.ndim > 1 else 1

    # Get actions for H2 calculation
    if 'actions' in data:
        actions = np.array(data['actions'])
        result['h2'] = compute_h2_diversity(actions)
        result['n_actions'] = len(actions)
        result['action_dim'] = actions.shape[1] if actions.ndim > 1 else 1

    # Get observations
    if 'observations' in data:
        obs = np.array(data['observations'])
        result['n_obs'] = len(obs)

    # Get metadata
    if 'metadata' in data:
        meta = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
        if isinstance(meta, dict):
            result['metadata'] = meta

    # Compute approximate Full K (with what we have)
    h1 = result.get('h1', 0.5)
    h2 = result.get('h2', 0.5)
    # H3-H7 default to 0.5 since we don't have them
    result['approx_full_k'] = (h1 + h2 + 0.5 + 0.5 + 0.5 + 0.5 + 0.0) / 7.0

    return result


def main():
    log_dir = Path('logs')
    npz_files = sorted(log_dir.glob('track_b_*.npz'))

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  📊 Deep Track B Log Analysis                                 ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    if not npz_files:
        print("No Track B .npz files found in logs/")
        return

    print(f"Found {len(npz_files)} Track B log files\n")

    # Analyze all files
    open_loop = []
    controller_train = []
    controller_eval = []

    for filepath in npz_files:
        result = analyze_episode(filepath)

        if result['mode'] == 'open_loop':
            open_loop.append(result)
        elif result['mode'] == 'train':
            controller_train.append(result)
        else:
            controller_eval.append(result)

    # Print sample structure
    print("=== Sample Episode Structure ===\n")
    sample = open_loop[0] if open_loop else controller_eval[0]
    for key, val in sample.items():
        if key != 'metadata':
            print(f"  {key}: {val}")
    print()

    # Compare Open Loop vs Controller
    print("=== Comparison: Open Loop vs Controller ===\n")

    def stats(episodes, metric):
        vals = [ep[metric] for ep in episodes if metric in ep]
        if vals:
            return np.mean(vals), np.std(vals)
        return 0, 0

    print("Metric           │ Open Loop (n={}) │ Controller Eval (n={})".format(
        len(open_loop), len(controller_eval)))
    print("─────────────────┼───────────────────┼────────────────────────")

    for metric in ['simple_k_mean', 'h1', 'h2', 'approx_full_k']:
        ol_mean, ol_std = stats(open_loop, metric)
        ce_mean, ce_std = stats(controller_eval, metric)

        if ol_mean > 0:
            change = ((ce_mean / ol_mean) - 1) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"

        print(f"{metric:16s} │ {ol_mean:6.4f} ± {ol_std:.3f}  │ {ce_mean:6.4f} ± {ce_std:.3f}  ({change_str})")

    # Key findings
    print("\n" + "=" * 70)
    print("\n🔑 KEY FINDINGS:\n")

    ol_simple_k = stats(open_loop, 'simple_k_mean')[0]
    ce_simple_k = stats(controller_eval, 'simple_k_mean')[0]

    if ol_simple_k > 0 and ce_simple_k > 0:
        k_improvement = ((ce_simple_k / ol_simple_k) - 1) * 100
        print(f"  Simple K improvement: {k_improvement:+.1f}%")

        if k_improvement > 50:
            print("  ⚠️  This is likely the '63% improvement' claim!")
            print("     But Simple K ANTI-CORRELATES with performance.")

    ol_h2 = stats(open_loop, 'h2')[0]
    ce_h2 = stats(controller_eval, 'h2')[0]

    if ol_h2 > 0 and ce_h2 > 0:
        h2_change = ((ce_h2 / ol_h2) - 1) * 100
        print(f"\n  H2 (Diversity) change: {h2_change:+.1f}%")

        if h2_change < 0:
            print("  ❌ Controller REDUCES diversity!")
            print("     This is the opposite of what we want.")
        elif h2_change > 10:
            print("  ✅ Controller increases diversity - good!")

    # Approximate Full K
    ol_full = stats(open_loop, 'approx_full_k')[0]
    ce_full = stats(controller_eval, 'approx_full_k')[0]

    if ol_full > 0 and ce_full > 0:
        full_change = ((ce_full / ol_full) - 1) * 100
        print(f"\n  Approx Full K change: {full_change:+.1f}%")
        print("  (Based on H1 + H2 + defaults for H3-H7)")

    print("\n" + "=" * 70)
    print("\n📋 CONCLUSION:\n")

    print("  The Track B logs show:")
    print("  - Simple K values are stored as 'K_values'")
    print("  - We can compute H1 from states, H2 from actions")
    print("  - H3-H7 need simulation data we don't have")
    print()
    print("  To properly validate Paper 1:")
    print("  1. Re-run Track B with Full K as reward signal")
    print("  2. Or add Full K computation to UniverseSimulator")
    print("  3. Then compare controller vs open-loop Full K")
    print()


if __name__ == '__main__':
    main()
