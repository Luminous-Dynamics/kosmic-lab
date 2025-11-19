#!/usr/bin/env python3
"""
Analyze Track B Logs with Full K-Index

Load existing Track B experiment logs and compute Full K-Index to see
if the "63% improvement" claim holds with the correct metric.
"""

import numpy as np
from pathlib import Path
from collections import Counter
import json


def compute_h2_diversity(actions):
    """H2: Action distribution entropy."""
    if len(actions) < 5:
        return 0.5
    counts = Counter(actions)
    if len(counts) <= 1:
        return 0.0
    probs = np.array([c / len(actions) for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    return float(h / np.log(len(counts)))


def compute_simple_k(obs_norms, act_norms):
    """Simple K-Index (deprecated)."""
    if len(obs_norms) < 2:
        return 0.0
    try:
        r = np.corrcoef(obs_norms, act_norms)[0, 1]
        return 2.0 * abs(r) if not np.isnan(r) else 0.0
    except:
        return 0.0


def analyze_npz(filepath):
    """Analyze a single Track B .npz file."""
    data = np.load(filepath, allow_pickle=True)

    result = {
        'file': filepath.name,
        'keys': list(data.keys())
    }

    # Try to extract K values
    if 'K' in data:
        k_values = data['K']
        result['avg_k'] = float(np.mean(k_values))
        result['final_k'] = float(k_values[-1]) if len(k_values) > 0 else 0
        result['k_values'] = k_values

    # Check for harmony values
    for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']:
        if h in data:
            result[f'{h}_mean'] = float(np.mean(data[h]))

    # Check for actions
    if 'actions' in data:
        actions = data['actions']
        if len(actions.shape) > 1:
            actions = actions.flatten()
        result['h2_computed'] = compute_h2_diversity(actions.tolist())

    return result


def main():
    log_dir = Path('logs')
    npz_files = sorted(log_dir.glob('track_b_*.npz'))

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  📊 Track B Log Analysis with Full K-Index                    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    if not npz_files:
        print("No Track B .npz files found in logs/")
        return

    print(f"Found {len(npz_files)} Track B log files\n")

    # Analyze first file to see structure
    print("=== Sample File Structure ===\n")
    sample = analyze_npz(npz_files[0])
    print(f"File: {sample['file']}")
    print(f"Keys: {sample['keys']}")
    print()

    # Categorize episodes
    open_loop = []
    controller_train = []
    controller_eval = []

    for filepath in npz_files:
        result = analyze_npz(filepath)

        if 'open_loop' in filepath.name:
            open_loop.append(result)
        elif 'controller_train' in filepath.name:
            controller_train.append(result)
        elif 'controller_eval' in filepath.name:
            controller_eval.append(result)

    print("=== Episode Counts ===\n")
    print(f"  Open loop:        {len(open_loop)}")
    print(f"  Controller train: {len(controller_train)}")
    print(f"  Controller eval:  {len(controller_eval)}")
    print()

    # Compare K values
    if open_loop and controller_eval:
        print("=== K-Index Comparison ===\n")

        if 'avg_k' in open_loop[0]:
            open_loop_ks = [ep['avg_k'] for ep in open_loop if 'avg_k' in ep]
            controller_ks = [ep['avg_k'] for ep in controller_eval if 'avg_k' in ep]

            if open_loop_ks and controller_ks:
                ol_mean = np.mean(open_loop_ks)
                ctrl_mean = np.mean(controller_ks)
                improvement = ((ctrl_mean / ol_mean) - 1) * 100 if ol_mean > 0 else 0

                print(f"  Open loop avg K:    {ol_mean:.4f}")
                print(f"  Controller avg K:   {ctrl_mean:.4f}")
                print(f"  Improvement:        {improvement:+.1f}%")
                print()

                if improvement > 50:
                    print("  ⚠️  This is the '63% improvement' claim!")
                    print("     But Simple K anti-correlates with performance.")
                    print("     Higher K = more rigidity, not better coherence.")

        # Check H2 if available
        if 'h2_computed' in open_loop[0]:
            ol_h2 = [ep['h2_computed'] for ep in open_loop if 'h2_computed' in ep]
            ctrl_h2 = [ep['h2_computed'] for ep in controller_eval if 'h2_computed' in ep]

            if ol_h2 and ctrl_h2:
                print("\n=== H2 (Diversity) Comparison ===\n")
                print(f"  Open loop H2:    {np.mean(ol_h2):.4f}")
                print(f"  Controller H2:   {np.mean(ctrl_h2):.4f}")
                h2_change = ((np.mean(ctrl_h2) / np.mean(ol_h2)) - 1) * 100 if np.mean(ol_h2) > 0 else 0
                print(f"  Change:          {h2_change:+.1f}%")

                if h2_change < 0:
                    print("\n  ❌ H2 DECREASED - controller reduces diversity!")
                    print("     This confirms optimization for rigidity, not coherence.")

    # Check for logged harmonies
    if open_loop and 'H1_mean' in open_loop[0]:
        print("\n=== Full 7-Harmony Analysis ===\n")

        for ep_type, episodes in [('Open Loop', open_loop), ('Controller', controller_eval)]:
            print(f"  {ep_type}:")
            for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']:
                key = f'{h}_mean'
                if key in episodes[0]:
                    vals = [ep[key] for ep in episodes if key in ep]
                    if vals:
                        print(f"    {h}: {np.mean(vals):.4f}")
            print()

    print("\n=== Conclusion ===\n")
    print("  To properly evaluate Track B, we need to:")
    print("  1. Check if the data contains all 7 harmonies")
    print("  2. Compute Full K = mean(H1-H7) from logs")
    print("  3. See if Full K improves or degrades with controller")
    print("  4. Check H2 (diversity) specifically")
    print()


if __name__ == '__main__':
    main()
