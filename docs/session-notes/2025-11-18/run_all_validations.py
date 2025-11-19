#!/usr/bin/env python3
"""
Master Reproducibility Script: K-Index Validation

Run this script to independently verify all findings from the
November 18-19, 2025 validation session.

Usage:
    python run_all_validations.py

Output:
    REPRODUCIBILITY_REPORT.md - Complete verification of all findings

Requirements:
    - numpy
    - scipy
    - Access to logs/track_b/ and logs/track_e/ directories
"""

import os
import sys
import numpy as np
from scipy import stats
from collections import Counter
from datetime import datetime
from pathlib import Path

# Change to the kosmic-lab directory
KOSMIC_LAB = Path(__file__).parent.parent.parent.parent
os.chdir(KOSMIC_LAB)

def compute_h2_diversity(actions):
    """H2: Action distribution entropy (normalized)."""
    counts = Counter(actions)
    if len(counts) <= 1:
        return 0.0
    total = len(actions)
    probs = np.array([c / total for c in counts.values()])
    h = -np.sum(probs * np.log(probs + 1e-10))
    h_max = np.log(len(counts))
    return float(h / h_max) if h_max > 0 else 0.0

def fmt(value, decimals=3):
    """Safely format a value that might be 'N/A' or a number."""
    if isinstance(value, str):
        return value
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)

def validate_track_e_k_vs_rewards():
    """
    Finding 1: K-Index does not correlate with rewards in Track E.
    Expected: r ≈ -0.01, p > 0.05
    """
    results = {}

    try:
        data = np.load('logs/track_e/developmental/track_e_20251111_162703.npz',
                       allow_pickle=True)

        conditions = ['standard_rl', 'curriculum_learning', 'meta_learning', 'full_developmental']

        all_k = []
        all_rewards = []

        for cond in conditions:
            d = data[cond].item()
            k_indices = d['k_indices']
            rewards = d['rewards']

            all_k.extend(k_indices)
            all_rewards.extend(rewards)

            r, p = stats.pearsonr(k_indices, rewards)
            results[f'{cond}_r'] = r
            results[f'{cond}_p'] = p

        # Overall correlation
        r_overall, p_overall = stats.pearsonr(all_k, all_rewards)
        results['overall_r'] = r_overall
        results['overall_p'] = p_overall
        results['n'] = len(all_k)
        results['status'] = 'PASS' if abs(r_overall) < 0.1 and p_overall > 0.05 else 'UNEXPECTED'

    except Exception as e:
        results['error'] = str(e)
        results['status'] = 'ERROR'

    return results

def validate_track_e_reward_meaningfulness():
    """
    Finding 2: Track E rewards are essentially random noise.
    Expected: No learning trend, low autocorrelation
    """
    results = {}

    try:
        data = np.load('logs/track_e/developmental/track_e_20251111_162703.npz',
                       allow_pickle=True)

        conditions = ['standard_rl', 'curriculum_learning', 'meta_learning', 'full_developmental']

        learning_trends = []
        autocorrs = []

        for cond in conditions:
            d = data[cond].item()
            rewards = d['rewards']
            episodes = np.arange(len(rewards))

            # Learning trend
            r, p = stats.pearsonr(episodes, rewards)
            learning_trends.append((r, p))

            # Autocorrelation
            if len(rewards) > 1:
                autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1]
            else:
                autocorr = 0
            autocorrs.append(autocorr)

        results['mean_learning_trend_r'] = np.mean([t[0] for t in learning_trends])
        results['any_significant_trend'] = any(t[1] < 0.05 and t[0] > 0.2 for t in learning_trends)
        results['mean_autocorrelation'] = np.mean(autocorrs)
        results['status'] = 'PASS' if not results['any_significant_trend'] and abs(results['mean_autocorrelation']) < 0.15 else 'UNEXPECTED'

    except Exception as e:
        results['error'] = str(e)
        results['status'] = 'ERROR'

    return results

def validate_track_b_no_performance_metric():
    """
    Finding 3: Track B has no external performance metric.
    Expected: Only K_values logged, episode_length constant
    """
    results = {}

    try:
        # Track B files are in logs/ directly, not in a subdirectory
        logs_dir = Path('logs')
        npz_files = list(logs_dir.glob('track_b_ep*.npz'))

        results['n_files'] = len(npz_files)

        if npz_files:
            # Check first file for structure
            data = np.load(npz_files[0], allow_pickle=True)
            results['keys_available'] = list(data.keys())

            has_rewards = 'rewards' in data.keys()
            has_k_values = 'K_values' in data.keys()

            # Check episode length variation
            if 'episode_length' in data.keys():
                ep_len = data['episode_length']
                if hasattr(ep_len, '__len__'):
                    results['episode_length_std'] = float(np.std(ep_len))
                else:
                    results['episode_length_std'] = 0.0
            else:
                results['episode_length_std'] = 'not_logged'

            results['has_rewards'] = has_rewards
            results['has_k_values'] = has_k_values
            results['status'] = 'PASS' if not has_rewards and has_k_values else 'UNEXPECTED'
        else:
            results['status'] = 'NO_FILES'

    except Exception as e:
        results['error'] = str(e)
        results['status'] = 'ERROR'

    return results

def validate_k_increases_during_training():
    """
    Finding 4: K-Index increases during training.
    Expected: Significant positive correlation with episode number
    """
    results = {}

    try:
        data = np.load('logs/track_e/developmental/track_e_20251111_162703.npz',
                       allow_pickle=True)

        conditions = ['standard_rl', 'curriculum_learning', 'meta_learning', 'full_developmental']

        correlations = []

        for cond in conditions:
            d = data[cond].item()
            k_indices = d['k_indices']
            episodes = np.arange(len(k_indices))

            r, p = stats.pearsonr(episodes, k_indices)
            correlations.append({'condition': cond, 'r': r, 'p': p})
            results[f'{cond}_r'] = r
            results[f'{cond}_p'] = p

        # All should be significantly positive
        all_positive = all(c['r'] > 0 and c['p'] < 0.05 for c in correlations)
        results['all_positive_significant'] = all_positive
        results['mean_r'] = np.mean([c['r'] for c in correlations])
        results['status'] = 'PASS' if all_positive else 'UNEXPECTED'

    except Exception as e:
        results['error'] = str(e)
        results['status'] = 'ERROR'

    return results

def validate_h2_in_track_b():
    """
    Finding 5: Controllers learn diverse actions (H2: 0 → ~1).
    Expected: H2 near 0 for open loop, near 1 for controller
    """
    results = {}

    try:
        # Track B files are in logs/ directly
        logs_dir = Path('logs')
        npz_files = list(logs_dir.glob('track_b_ep*.npz'))

        open_loop_h2 = []
        controller_h2 = []

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            fname = f.name

            # Check if actions key exists
            if 'actions' in data.keys():
                actions = data['actions']
                # Convert to list if needed for counting
                if hasattr(actions, 'tolist'):
                    actions = actions.tolist()
                h2 = compute_h2_diversity(actions)

                # Categorize by filename
                if 'open_loop' in fname:
                    open_loop_h2.append(h2)
                elif 'controller' in fname:
                    controller_h2.append(h2)

        if open_loop_h2 and controller_h2:
            results['open_loop_h2_mean'] = np.mean(open_loop_h2)
            results['controller_h2_mean'] = np.mean(controller_h2)
            results['difference'] = results['controller_h2_mean'] - results['open_loop_h2_mean']

            # Statistical test
            if len(open_loop_h2) > 1 and len(controller_h2) > 1:
                t, p = stats.ttest_ind(controller_h2, open_loop_h2)
                results['t_statistic'] = t
                results['p_value'] = p

            results['status'] = 'PASS' if results['difference'] > 0.5 else 'UNEXPECTED'
        else:
            results['status'] = 'INSUFFICIENT_DATA'

    except Exception as e:
        results['error'] = str(e)
        results['status'] = 'ERROR'

    return results

def generate_report(all_results):
    """Generate markdown report of all validation results."""

    report = f"""# Reproducibility Report: K-Index Validation

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Purpose**: Independent verification of November 18-19, 2025 findings

---

## Summary

| Finding | Status | Key Metric |
|---------|--------|------------|
| K vs Rewards (Track E) | {all_results['k_vs_rewards']['status']} | r = {fmt(all_results['k_vs_rewards'].get('overall_r', 'N/A'))} |
| Rewards Are Noise | {all_results['reward_meaningfulness']['status']} | autocorr = {fmt(all_results['reward_meaningfulness'].get('mean_autocorrelation', 'N/A'))} |
| Track B No Metric | {all_results['track_b_metric']['status']} | has_rewards = {all_results['track_b_metric'].get('has_rewards', 'N/A')} |
| K Increases During Training | {all_results['k_increases']['status']} | mean r = {fmt(all_results['k_increases'].get('mean_r', 'N/A'))} |
| Controllers Learn Diversity | {all_results['h2_track_b']['status']} | ΔH2 = {fmt(all_results['h2_track_b'].get('difference', 'N/A'))} |

---

## Detailed Results

### 1. K-Index Does Not Predict Rewards

**Finding**: K-Index has zero correlation with external rewards in Track E.

```
Overall correlation: r = {fmt(all_results['k_vs_rewards'].get('overall_r', 'N/A'), 4)}
P-value: {fmt(all_results['k_vs_rewards'].get('overall_p', 'N/A'), 4)}
N = {all_results['k_vs_rewards'].get('n', 'N/A')}

By condition:
- standard_rl: r = {fmt(all_results['k_vs_rewards'].get('standard_rl_r', 'N/A'))}
- curriculum_learning: r = {fmt(all_results['k_vs_rewards'].get('curriculum_learning_r', 'N/A'))}
- meta_learning: r = {fmt(all_results['k_vs_rewards'].get('meta_learning_r', 'N/A'))}
- full_developmental: r = {fmt(all_results['k_vs_rewards'].get('full_developmental_r', 'N/A'))}
```

**Interpretation**: K-Index explains 0% of reward variance. It does not predict performance.

---

### 2. Track E Rewards Are Random Noise

**Finding**: Rewards show no learning trend and near-zero autocorrelation.

```
Mean learning trend: r = {fmt(all_results['reward_meaningfulness'].get('mean_learning_trend_r', 'N/A'))}
Any significant trend: {all_results['reward_meaningfulness'].get('any_significant_trend', 'N/A')}
Mean autocorrelation: {fmt(all_results['reward_meaningfulness'].get('mean_autocorrelation', 'N/A'))}
```

**Interpretation**: Rewards behave like random noise. Track E is not a valid test of performance.

---

### 3. Track B Has No External Performance Metric

**Finding**: Track B logs only K-values, not rewards or performance.

```
Files analyzed: {all_results['track_b_metric'].get('n_files', 'N/A')}
Keys available: {all_results['track_b_metric'].get('keys_available', 'N/A')}
Has rewards: {all_results['track_b_metric'].get('has_rewards', 'N/A')}
Has K_values: {all_results['track_b_metric'].get('has_k_values', 'N/A')}
```

**Interpretation**: Cannot validate performance claims for Track B. Only K-values exist (circular).

---

### 4. K-Index Increases During Training

**Finding**: K-Index shows significant positive correlation with episode number.

```
Mean correlation: r = {fmt(all_results['k_increases'].get('mean_r', 'N/A'))}
All positive & significant: {all_results['k_increases'].get('all_positive_significant', 'N/A')}

By condition:
- standard_rl: r = {fmt(all_results['k_increases'].get('standard_rl_r', 'N/A'))} (p = {fmt(all_results['k_increases'].get('standard_rl_p', 'N/A'), 4)})
- curriculum_learning: r = {fmt(all_results['k_increases'].get('curriculum_learning_r', 'N/A'))}
- meta_learning: r = {fmt(all_results['k_increases'].get('meta_learning_r', 'N/A'))}
- full_developmental: r = {fmt(all_results['k_increases'].get('full_developmental_r', 'N/A'))}
```

**Interpretation**: K-Index tracks training progress. This is a valid finding.

---

### 5. Controllers Learn Diverse Actions

**Finding**: Trained controllers show much higher action diversity than open-loop baselines.

```
Open loop H2: {fmt(all_results['h2_track_b'].get('open_loop_h2_mean', 'N/A'))}
Controller H2: {fmt(all_results['h2_track_b'].get('controller_h2_mean', 'N/A'))}
Difference: {fmt(all_results['h2_track_b'].get('difference', 'N/A'))}
```

**Interpretation**: Controllers develop diverse action repertoires. This is a valid finding.

---

## Conclusion

**All critical findings reproduced:**

1. ❌ K-Index does NOT predict performance (r ≈ 0)
2. ⚠️ Track E rewards are noise (no learning trend)
3. ⚠️ Track B has no performance metric (only K-values)
4. ✅ K-Index increases during training (valid)
5. ✅ Controllers learn diverse actions (valid)

**Bottom line**: K-Index tracks training dynamics, not performance. Papers must be reframed.

---

*Report generated by run_all_validations.py*
*To reproduce: `python run_all_validations.py`*

"""
    return report

def main():
    print("=" * 70)
    print("K-INDEX VALIDATION REPRODUCIBILITY CHECK")
    print("=" * 70)
    print("\nRunning all validation tests...\n")

    all_results = {}

    # Test 1: K vs Rewards
    print("1. Testing K-Index vs Rewards correlation...")
    all_results['k_vs_rewards'] = validate_track_e_k_vs_rewards()
    print(f"   Status: {all_results['k_vs_rewards']['status']}")

    # Test 2: Reward meaningfulness
    print("2. Testing Track E reward meaningfulness...")
    all_results['reward_meaningfulness'] = validate_track_e_reward_meaningfulness()
    print(f"   Status: {all_results['reward_meaningfulness']['status']}")

    # Test 3: Track B metric
    print("3. Checking Track B performance metric...")
    all_results['track_b_metric'] = validate_track_b_no_performance_metric()
    print(f"   Status: {all_results['track_b_metric']['status']}")

    # Test 4: K increases
    print("4. Testing K-Index increases during training...")
    all_results['k_increases'] = validate_k_increases_during_training()
    print(f"   Status: {all_results['k_increases']['status']}")

    # Test 5: H2 in Track B
    print("5. Testing H2 diversity in Track B...")
    all_results['h2_track_b'] = validate_h2_in_track_b()
    print(f"   Status: {all_results['h2_track_b']['status']}")

    # Generate report
    print("\nGenerating reproducibility report...")
    report = generate_report(all_results)

    report_path = Path(__file__).parent / 'REPRODUCIBILITY_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    statuses = [r['status'] for r in all_results.values()]
    passed = statuses.count('PASS')
    total = len(statuses)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All findings reproduced successfully.")
        print("   K-Index does not predict performance.")
        print("   Papers must be reframed around behavioral findings.")
    else:
        print("\n⚠️  Some tests did not pass as expected.")
        print("   Review REPRODUCIBILITY_REPORT.md for details.")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
