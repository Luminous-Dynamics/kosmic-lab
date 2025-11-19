#!/usr/bin/env python3
"""
Validate H2 in Actual Track B Experimental Logs

This is the critical test: Does H2 (Diversity) correlate with performance
in the ACTUAL Track B data, not CartPole simulations?

Key questions:
1. What performance metrics exist in the logs?
2. Does H2 correlate with them?
3. Does this validate or invalidate our CartPole findings?
"""

import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_h2_diversity(actions):
    """H2: Action distribution entropy."""
    if len(actions) < 5:
        return np.nan

    # Handle multi-dimensional actions
    if isinstance(actions, np.ndarray) and actions.ndim > 1:
        # Bin continuous actions
        flat = actions.flatten()
        if len(flat) < 5:
            return np.nan
        bins = np.linspace(flat.min() - 1e-10, flat.max() + 1e-10, 11)
        discretized = np.digitize(flat, bins)
    else:
        discretized = np.array(actions).flatten()

    counts = Counter(discretized)
    if len(counts) <= 1:
        return 0.0

    total = len(discretized)
    probs = np.array([c / total for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    h_max = np.log(len(counts))

    return float(h / h_max) if h_max > 0 else 0.0

def compute_simple_k(observations, actions):
    """Simple K-Index for comparison."""
    if len(observations) < 5 or len(actions) < 5:
        return np.nan

    obs = np.array(observations)
    act = np.array(actions)

    # Align lengths
    min_len = min(len(obs), len(act))
    obs = obs[:min_len]
    act = act[:min_len]

    # Compute norms
    if obs.ndim > 1:
        obs_norms = np.linalg.norm(obs, axis=1)
    else:
        obs_norms = np.abs(obs)

    if act.ndim > 1:
        act_norms = np.linalg.norm(act, axis=1)
    else:
        act_norms = np.abs(act)

    if np.std(obs_norms) < 1e-10 or np.std(act_norms) < 1e-10:
        return 0.0

    r = np.corrcoef(obs_norms, act_norms)[0, 1]
    if np.isnan(r):
        return 0.0

    return float(2 * abs(r))

def analyze_log_file(filepath):
    """Analyze a single Track B log file."""
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        return None

    result = {
        'file': filepath.name,
        'mode': 'open_loop' if 'open_loop' in filepath.name else
                'train' if 'train' in filepath.name else 'eval'
    }

    # Extract available data
    keys = list(data.keys())
    result['available_keys'] = keys

    # Actions
    if 'actions' in data:
        actions = np.array(data['actions'])
        result['n_actions'] = len(actions)
        result['action_shape'] = actions.shape
        result['h2'] = compute_h2_diversity(actions)
    else:
        result['h2'] = np.nan

    # Observations
    if 'observations' in data:
        observations = np.array(data['observations'])
        result['n_observations'] = len(observations)
        result['obs_shape'] = observations.shape
    else:
        observations = None

    # States
    if 'states' in data:
        states = np.array(data['states'])
        result['n_states'] = len(states)
        result['state_shape'] = states.shape

    # Simple K
    if 'actions' in data and observations is not None:
        result['simple_k'] = compute_simple_k(observations, actions)
    else:
        result['simple_k'] = np.nan

    # K values from log
    if 'K_values' in data:
        k_values = np.array(data['K_values']).flatten()
        result['k_mean'] = float(np.mean(k_values))
        result['k_final'] = float(k_values[-1])
        result['k_std'] = float(np.std(k_values))

    # Rewards if available
    if 'rewards' in data:
        rewards = np.array(data['rewards']).flatten()
        result['total_reward'] = float(np.sum(rewards))
        result['mean_reward'] = float(np.mean(rewards))

    # Episode length as performance proxy
    if 'actions' in data:
        result['episode_length'] = len(data['actions'])

    return result

def main():
    print("=" * 70)
    print("VALIDATING H2 IN ACTUAL TRACK B EXPERIMENTAL LOGS")
    print("=" * 70)
    print("\nThis is the critical test: Does H2 correlate with performance")
    print("in the ACTUAL experimental data?\n")

    log_dir = Path('logs')
    npz_files = sorted(log_dir.glob('track_b_*.npz'))

    if not npz_files:
        print("ERROR: No Track B .npz files found!")
        return

    print(f"Found {len(npz_files)} Track B log files\n")

    # Analyze all files
    results = []
    for filepath in npz_files:
        result = analyze_log_file(filepath)
        if result:
            results.append(result)

    print(f"Successfully analyzed {len(results)} files\n")

    # Show what data is available
    if results:
        print("=" * 70)
        print("DATA AVAILABLE IN LOGS")
        print("=" * 70)
        keys = set()
        for r in results:
            if 'available_keys' in r:
                keys.update(r['available_keys'])
        print(f"\nKeys found: {sorted(keys)}")

        # Show sample
        sample = results[0]
        print(f"\nSample file: {sample['file']}")
        if 'action_shape' in sample:
            print(f"  Action shape: {sample['action_shape']}")
        if 'obs_shape' in sample:
            print(f"  Observation shape: {sample['obs_shape']}")
        if 'state_shape' in sample:
            print(f"  State shape: {sample['state_shape']}")

    # Separate by mode
    open_loop = [r for r in results if r['mode'] == 'open_loop']
    controller = [r for r in results if r['mode'] in ['train', 'eval']]

    print(f"\nOpen loop episodes: {len(open_loop)}")
    print(f"Controller episodes: {len(controller)}")

    # Compute H2 statistics
    print("\n" + "=" * 70)
    print("H2 (DIVERSITY) BY CONDITION")
    print("=" * 70)

    ol_h2 = [r['h2'] for r in open_loop if not np.isnan(r.get('h2', np.nan))]
    ctrl_h2 = [r['h2'] for r in controller if not np.isnan(r.get('h2', np.nan))]

    if ol_h2:
        print(f"\nOpen Loop H2: {np.mean(ol_h2):.4f} ± {np.std(ol_h2):.4f}")
    if ctrl_h2:
        print(f"Controller H2: {np.mean(ctrl_h2):.4f} ± {np.std(ctrl_h2):.4f}")

    if ol_h2 and ctrl_h2:
        t_stat, p_val = stats.ttest_ind(ol_h2, ctrl_h2)
        print(f"\nDifference: p = {p_val:.4f}")
        if p_val < 0.05:
            print("✅ Controller has significantly different H2 than open loop")
        else:
            print("⚠️ No significant difference in H2")

    # Now the critical question: What is "performance"?
    print("\n" + "=" * 70)
    print("IDENTIFYING PERFORMANCE METRIC")
    print("=" * 70)

    # Check what metrics vary across episodes
    potential_metrics = ['k_mean', 'k_final', 'total_reward', 'episode_length']

    for metric in potential_metrics:
        values = [r[metric] for r in results if metric in r and not np.isnan(r.get(metric, np.nan))]
        if values:
            print(f"\n{metric}:")
            print(f"  Range: [{min(values):.3f}, {max(values):.3f}]")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std: {np.std(values):.3f}")
            if np.std(values) < 1e-6:
                print(f"  ⚠️ No variation - cannot use as performance metric")

    # Correlate H2 with available metrics
    print("\n" + "=" * 70)
    print("H2 CORRELATION WITH AVAILABLE METRICS")
    print("=" * 70)

    # Combine all results for correlation
    all_h2 = []
    all_metrics = {m: [] for m in potential_metrics}

    for r in results:
        h2 = r.get('h2', np.nan)
        if np.isnan(h2):
            continue

        all_h2.append(h2)
        for m in potential_metrics:
            all_metrics[m].append(r.get(m, np.nan))

    print("\nMetric          | Correlation with H2 | p-value | Interpretation")
    print("-" * 70)

    for metric in potential_metrics:
        values = all_metrics[metric]
        valid_pairs = [(h, v) for h, v in zip(all_h2, values) if not np.isnan(v)]

        if len(valid_pairs) < 5:
            print(f"{metric:15s} | N/A (insufficient data)")
            continue

        h2_valid = [p[0] for p in valid_pairs]
        metric_valid = [p[1] for p in valid_pairs]

        if np.std(metric_valid) < 1e-10:
            print(f"{metric:15s} | N/A (no variation)")
            continue

        r, p = stats.pearsonr(h2_valid, metric_valid)

        if metric == 'k_mean' or metric == 'k_final':
            interp = "⚠️ Circular!" if abs(r) > 0.3 else "Weak"
        elif r > 0.3:
            interp = "✅ Positive"
        elif r < -0.3:
            interp = "❌ Negative"
        else:
            interp = "Weak"

        print(f"{metric:15s} | r = {r:+.3f}           | p = {p:.3f} | {interp}")

    # Critical analysis
    print("\n" + "=" * 70)
    print("CRITICAL ANALYSIS")
    print("=" * 70)

    # Check if K is the only "performance" metric
    has_reward = any('total_reward' in r for r in results)
    has_length = any('episode_length' in r and r['episode_length'] > 0 for r in results)

    if not has_reward and not has_length:
        print("\n⚠️ PROBLEM: No external performance metric found!")
        print("   Only K-Index values are available.")
        print("   Correlating H2 with K is CIRCULAR - both measure the agent's behavior.")
        print("\n   This means we CANNOT validate that H2 predicts 'performance'")
        print("   in Track B because there is no external performance metric.")
    else:
        if has_reward:
            print("\n✅ External metric found: reward")
        if has_length:
            print("\n✅ External metric found: episode_length")

    # Compare Simple K vs H2
    print("\n" + "=" * 70)
    print("SIMPLE K VS H2")
    print("=" * 70)

    ol_sk = [r['simple_k'] for r in open_loop if not np.isnan(r.get('simple_k', np.nan))]
    ctrl_sk = [r['simple_k'] for r in controller if not np.isnan(r.get('simple_k', np.nan))]

    if ol_sk and ctrl_sk:
        print(f"\nOpen Loop Simple K: {np.mean(ol_sk):.4f} ± {np.std(ol_sk):.4f}")
        print(f"Controller Simple K: {np.mean(ctrl_sk):.4f} ± {np.std(ctrl_sk):.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if not has_reward and not has_length:
        print("\n❌ CANNOT VALIDATE H2 IN TRACK B")
        print("\nReason: Track B logs contain no external performance metric.")
        print("The only available metrics (K_values) are what the agent optimizes for,")
        print("so correlating H2 with K would be circular reasoning.")
        print("\nTo validate H2, we need:")
        print("  1. Re-run Track B with external performance logging")
        print("  2. OR use Track E/F which may have task rewards")
        print("  3. OR acknowledge this limitation in papers")
    else:
        print("\n✅ External metrics available - correlation analysis valid")

    print("\n✅ Analysis complete\n")

if __name__ == '__main__':
    main()
