# 🎯 Track F Correction Guide - Surgical FGSM Fix

**Purpose**: Apply Phase 1 corrected FGSM to Track F and generate publication-ready statistics

**Status**: Ready to implement
**Estimated Time**: 2-3 hours (1 hour patches + 2 hours re-run)

---

## 🔧 Part 1: Minimal Runner Patches

### Patch 1: Add PyTorch imports and correct FGSM

**Location**: Top of `fre/track_f_runner.py` (after existing imports)

```python
# Add these imports
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

# Import Phase 1 modules
from fre.attacks.fgsm import fgsm_observation, sanity_check_loss_increases
from fre.metrics.k_index import k_index, k_index_robust
```

### Patch 2: Create PyTorch policy wrapper

**Location**: After `RobustLearner` class definition (around line 114)

```python
class TorchPolicyWrapper(nn.Module):
    """PyTorch wrapper for numpy policy (for FGSM gradient computation)."""

    def __init__(self, numpy_weights: np.ndarray):
        super().__init__()
        # Convert numpy weights to torch parameter
        self.fc = nn.Linear(numpy_weights.shape[1], numpy_weights.shape[0], bias=False)
        self.fc.weight.data = torch.from_numpy(numpy_weights).float()

    def forward(self, x):
        """Forward pass: linear + tanh."""
        return torch.tanh(self.fc(x))

    def sync_from_numpy(self, numpy_weights: np.ndarray):
        """Update torch weights from numpy weights."""
        self.fc.weight.data = torch.from_numpy(numpy_weights).float()
```

### Patch 3: Replace incorrect FGSM in `apply_perturbation()` method

**Location**: `AdversarialEnvironment.apply_perturbation()` method (lines 102-106)

**OLD (INCORRECT)**:
```python
elif perturbation_type == "gradient_based":
    # Simple FGSM-style perturbation (sign of gradient approximation)
    # Approximate gradient as direction that increases magnitude
    gradient_approx = np.sign(data)
    perturbed = data + strength * gradient_approx
```

**NEW (CORRECT)**:
```python
elif perturbation_type == "gradient_based":
    # CORRECT FGSM: x' = x + ε × sign(∇_x L(x,y))
    # This requires PyTorch policy and loss computation
    # WARNING: This branch is called from within episode loop where we don't have
    # agent policy access. We need to refactor to pass policy through.
    # For now, mark that this needs correction:
    raise NotImplementedError(
        "FGSM must be applied at episode level with access to policy. "
        "See corrected implementation in run_episode_with_fgsm()."
    )
```

### Patch 4: Add corrected FGSM episode runner

**Location**: After `run_episode()` function (around line 220)

```python
def run_episode_with_fgsm(env: AdversarialEnvironment, agent: RobustLearner,
                          exploration_noise: float = 0.1,
                          fgsm_epsilon: float = 0.1,
                          log_sanity: bool = True) -> Dict:
    """
    Run episode with CORRECT FGSM perturbation.

    This applies FGSM at each step by:
    1. Converting numpy policy to PyTorch
    2. Computing gradient w.r.t. observation
    3. Applying ε × sign(gradient) perturbation
    """
    # Create PyTorch policy wrapper
    torch_policy = TorchPolicyWrapper(agent.policy_weights)
    torch_policy.eval()

    # Loss function: MSE between action and "ideal" action (state alignment)
    loss_fn = nn.MSELoss()

    obs = env.reset()
    done = False
    episode_reward = 0.0
    k_indices = []

    # Sanity check tracking
    sanity_checks = []

    while not done:
        # Convert observation to torch tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # [1, obs_dim]

        # Apply FGSM if this is gradient_based condition
        if env.condition.perturbation_type == "gradient_based":
            if np.random.random() < env.condition.perturbation_frequency:
                # Sync torch policy with latest numpy weights
                torch_policy.sync_from_numpy(agent.policy_weights)

                # Compute "ideal" target action (env state alignment)
                # Target: action that aligns with state
                with torch.no_grad():
                    target_direction = torch.from_numpy(
                        env.state[:agent.action_dim] / (np.linalg.norm(env.state[:agent.action_dim]) + 1e-8)
                    ).float().unsqueeze(0)

                # Sanity check BEFORE applying FGSM
                if log_sanity:
                    base_loss, adv_loss = sanity_check_loss_increases(
                        torch_policy, obs_tensor, target_direction, loss_fn, fgsm_epsilon
                    )
                    sanity_checks.append({
                        'step': len(agent.obs_history),
                        'base_loss': base_loss,
                        'adv_loss': adv_loss,
                        'increased': adv_loss >= base_loss
                    })

                # Apply CORRECT FGSM
                obs_tensor = fgsm_observation(
                    torch_policy, obs_tensor, target_direction, loss_fn, fgsm_epsilon
                )

        # Convert back to numpy
        obs_perturbed = obs_tensor.squeeze(0).numpy()

        # Agent acts on (possibly perturbed) observation
        action = agent.act(obs_perturbed, exploration_noise)

        # Environment steps (without FGSM, since we already applied it)
        next_obs, reward, done = env.step(action)

        # Agent updates
        agent.update(reward)

        # Track metrics
        episode_reward += reward
        k_index = agent.get_k_index()
        k_indices.append(k_index)

        obs = next_obs

    # Final metrics with robust K-Index variants
    obs_norms = np.linalg.norm(np.array(agent.obs_history[-100:]), axis=1)
    act_norms = np.linalg.norm(np.array(agent.action_history[-100:]), axis=1)

    k_robust = k_index_robust(obs_norms, act_norms)

    return {
        'final_k': agent.get_k_index(),
        'mean_k': np.mean(k_indices) if len(k_indices) > 0 else 0.0,
        'k_variance': np.var(k_indices) if len(k_indices) > 0 else 0.0,
        'episode_reward': episode_reward,
        'k_history': k_indices,
        'k_pearson': k_robust['k_pearson'],
        'k_pearson_z': k_robust['k_pearson_z'],
        'k_spearman': k_robust['k_spearman'],
        'sanity_checks': sanity_checks if len(sanity_checks) > 0 else None
    }
```

### Patch 5: Update `run_condition()` to use corrected runner and log results

**Location**: Replace `run_condition()` function (lines 222-330)

```python
def run_condition(condition: AdversarialCondition, config: Dict,
                  baseline_k: Optional[float] = None) -> Dict:
    """Run all episodes for one adversarial condition with enhanced logging."""
    print(f"\n{'='*80}")
    print(f"Running condition: {condition.name}")
    print(f"Description: {condition.description}")
    print(f"Perturbation: {condition.perturbation_type}, strength={condition.perturbation_strength}, freq={condition.perturbation_frequency}")
    print(f"{'='*80}\n")

    # Extract configuration
    env_config = config['environment']
    agent_config = config['agent']
    n_episodes = config['experiment']['n_episodes']

    # Create environment and agent
    env = AdversarialEnvironment(
        obs_dim=env_config['obs_dim'],
        action_dim=env_config['action_dim'],
        max_steps=env_config['max_steps'],
        base_difficulty=env_config['base_difficulty'],
        condition=condition
    )

    agent = RobustLearner(
        obs_dim=env_config['obs_dim'],
        action_dim=env_config['action_dim'],
        learning_rate=agent_config['learning_rate']
    )

    # Episode-level logging
    episode_logs = []
    all_sanity_checks = []

    # Run episodes
    k_values = []
    rewards = []

    for episode_idx in range(n_episodes):
        # Use corrected FGSM runner if gradient_based, else standard
        if condition.perturbation_type == "gradient_based":
            metrics = run_episode_with_fgsm(
                env, agent,
                exploration_noise=agent_config['exploration_noise'],
                fgsm_epsilon=condition.perturbation_strength,
                log_sanity=True
            )
        else:
            metrics = run_episode(
                env, agent,
                exploration_noise=agent_config['exploration_noise']
            )
            # Add robust K-Index for non-FGSM conditions too
            if len(agent.obs_history) >= 10:
                obs_norms = np.linalg.norm(np.array(agent.obs_history[-100:]), axis=1)
                act_norms = np.linalg.norm(np.array(agent.action_history[-100:]), axis=1)
                k_robust = k_index_robust(obs_norms, act_norms)
                metrics.update({
                    'k_pearson': k_robust['k_pearson'],
                    'k_pearson_z': k_robust['k_pearson_z'],
                    'k_spearman': k_robust['k_spearman']
                })

        k_values.append(metrics['final_k'])
        rewards.append(metrics['episode_reward'])

        # Log per-episode data
        episode_log = {
            'condition': condition.name,
            'episode': episode_idx,
            'k': float(metrics['final_k']),
            'k_pearson': float(metrics.get('k_pearson', np.nan)),
            'k_pearson_z': float(metrics.get('k_pearson_z', np.nan)),
            'k_spearman': float(metrics.get('k_spearman', np.nan)),
            'reward_sum': float(metrics['episode_reward']),
            'k_variance': float(metrics['k_variance'])
        }
        episode_logs.append(episode_log)

        # Collect sanity checks if available
        if metrics.get('sanity_checks'):
            all_sanity_checks.extend(metrics['sanity_checks'])

        # Progress reporting
        if (episode_idx + 1) % 10 == 0:
            recent_k = np.mean(k_values[-10:])
            print(f"Episode {episode_idx + 1}/{n_episodes}: Recent K = {recent_k:.3f}")

    # Compute summary statistics
    k_array = np.array(k_values)
    mean_k = np.mean(k_array)
    std_k = np.std(k_array)

    # Robustness ratio if baseline provided
    robustness_ratio = (mean_k / baseline_k * 100) if baseline_k and baseline_k > 0 else None

    print(f"\n{condition.name} Summary:")
    print(f"  Mean K-Index: {mean_k:.4f} ± {std_k:.4f}")
    if robustness_ratio:
        print(f"  Robustness: {robustness_ratio:.1f}% of baseline")
    print(f"  Mean Reward: {np.mean(rewards):.4f}")

    # FGSM sanity check summary
    if len(all_sanity_checks) > 0:
        n_increased = sum(1 for sc in all_sanity_checks if sc['increased'])
        pct_increased = 100 * n_increased / len(all_sanity_checks)
        print(f"  FGSM Sanity: {pct_increased:.1f}% of steps increased loss ({n_increased}/{len(all_sanity_checks)})")

    return {
        'condition_name': condition.name,
        'k_values': k_values,
        'mean_k': mean_k,
        'std_k': std_k,
        'robustness_ratio': robustness_ratio,
        'episode_logs': episode_logs,
        'sanity_checks': all_sanity_checks if len(all_sanity_checks) > 0 else None
    }
```

### Patch 6: Update `main()` to save detailed logs

**Location**: In `main()` function, after running all conditions

```python
def main():
    # ... existing setup code ...

    # After running all conditions, save detailed logs
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate all episode logs
    all_episode_logs = []
    all_sanity_checks = []

    for result in results.values():
        all_episode_logs.extend(result['episode_logs'])
        if result.get('sanity_checks'):
            # Add condition name to sanity checks
            for sc in result['sanity_checks']:
                sc['condition'] = result['condition_name']
            all_sanity_checks.extend(result['sanity_checks'])

    # Save episode metrics (for analysis script)
    episode_df = pd.DataFrame(all_episode_logs)
    episode_df.to_csv(output_dir / "track_f_episode_metrics.csv", index=False)
    np.savez(output_dir / "track_f_episode_metrics.npz", data=np.array(all_episode_logs, dtype=object))
    print(f"\n✅ Saved episode metrics: {output_dir / 'track_f_episode_metrics.csv'}")

    # Save FGSM sanity checks (for verification)
    if len(all_sanity_checks) > 0:
        sanity_df = pd.DataFrame(all_sanity_checks)
        sanity_df.to_csv(output_dir / "fgsm_sanity_checks.csv", index=False)
        print(f"✅ Saved FGSM sanity checks: {output_dir / 'fgsm_sanity_checks.csv'}")

    # ... rest of existing main() code ...
```

---

## 📊 Part 2: Analysis Script

**File**: `fre/analyze_track_f.py` (new file)

```python
#!/usr/bin/env python3
"""
Track F Analysis Script - Generate publication-ready statistics.

Usage:
    python fre/analyze_track_f.py --input logs/track_f/track_f_episode_metrics.csv
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(vals, n=5000, alpha=0.05, rng=0):
    """Compute bootstrap confidence interval."""
    r = np.random.default_rng(rng)
    boots = [np.mean(r.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return (np.percentile(boots, 100*alpha/2), np.percentile(boots, 100*(1-alpha/2)))


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    m1, m2 = np.mean(a), np.mean(b)
    s = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) / (len(a)+len(b)-2))
    return (m1-m2)/s if s > 0 else np.nan


def analyze_track_f(csv_path: Path, output_dir: Path):
    """Generate complete Track F analysis."""

    # Load data
    df = pd.read_csv(csv_path)
    conds = df["condition"].unique()

    print("=" * 80)
    print("Track F Analysis - Publication Statistics")
    print("=" * 80)

    # 1. Summary Statistics Table
    print("\n📊 Summary Statistics (Mean ± SE, 95% CI)")
    print("-" * 80)

    rows = []
    for c in conds:
        k = df.loc[df.condition==c, "k"].values
        mean = k.mean()
        se = k.std(ddof=1)/np.sqrt(len(k))
        lo, hi = bootstrap_ci(k)
        rows.append([c, len(k), mean, se, lo, hi])

    summary = pd.DataFrame(rows, columns=["condition","n","mean_k","se","ci95_lo","ci95_hi"])
    print(summary.round(3).to_string(index=False))
    summary.to_csv(output_dir / "track_f_summary.csv", index=False)
    print(f"✅ Saved: {output_dir / 'track_f_summary.csv'}")

    # 2. Pairwise Comparisons vs Baseline
    print("\n📈 Pairwise Comparisons (vs Baseline)")
    print("-" * 80)

    baseline = df.loc[df.condition=="Baseline", "k"].values
    pairs, p_raw, ds, means = [], [], [], []

    for c in conds:
        if c == "Baseline":
            continue
        k = df.loc[df.condition==c, "k"].values
        _, p = stats.ttest_ind(k, baseline, equal_var=False)
        p_raw.append(p)
        pairs.append(("Baseline", c))
        ds.append(cohens_d(k, baseline))
        means.append((baseline.mean(), k.mean()))

    # FDR correction
    rej, p_fdr, _, _ = multipletests(p_raw, method="fdr_bh", alpha=0.05)

    comp = pd.DataFrame({
        "comparison": [f"{a} vs {b}" for a,b in pairs],
        "baseline_mean": [m[0] for m in means],
        "condition_mean": [m[1] for m in means],
        "cohens_d": ds,
        "p_raw": p_raw,
        "p_fdr": p_fdr,
        "significant": rej
    })

    print(comp.round(4).to_string(index=False))
    comp.to_csv(output_dir / "track_f_comparisons.csv", index=False)
    print(f"✅ Saved: {output_dir / 'track_f_comparisons.csv'}")

    # 3. FGSM Loss Sanity Check
    sanity_path = output_dir / "fgsm_sanity_checks.csv"
    if sanity_path.exists():
        print("\n🔍 FGSM Sanity Check (Loss Increase Verification)")
        print("-" * 80)

        sanity_df = pd.read_csv(sanity_path)
        fgsm_ok = sanity_df["increased"].mean()
        total = len(sanity_df)
        n_ok = sanity_df["increased"].sum()

        print(f"Loss increased in {fgsm_ok*100:.1f}% of FGSM steps ({n_ok}/{total})")
        print(f"Mean base loss: {sanity_df['base_loss'].mean():.5f}")
        print(f"Mean adv loss: {sanity_df['adv_loss'].mean():.5f}")

        if fgsm_ok < 0.95:
            print("⚠️  WARNING: <95% of steps increased loss. Check epsilon or policy.")
        else:
            print("✅ FGSM working correctly (>95% steps increase loss)")

    # 4. Manuscript-Ready Text
    print("\n📝 Manuscript Text Snippets")
    print("=" * 80)

    # Find adversarial condition
    adv_row = summary[summary.condition.str.contains("Adversarial")]
    base_row = summary[summary.condition == "Baseline"]

    if len(adv_row) > 0 and len(base_row) > 0:
        adv_mean = adv_row.iloc[0]["mean_k"]
        adv_se = adv_row.iloc[0]["se"]
        base_mean = base_row.iloc[0]["mean_k"]
        base_se = base_row.iloc[0]["se"]

        adv_comp = comp[comp.comparison.str.contains("Adversarial")]
        if len(adv_comp) > 0:
            d = adv_comp.iloc[0]["cohens_d"]
            p_fdr = adv_comp.iloc[0]["p_fdr"]

            ratio = (adv_mean / base_mean - 1) * 100

            print("\nResults — Adversarial Impact:")
            print(f'\"FGSM increased mean K-Index to {adv_mean:.2f} ± {adv_se:.2f} (SE) vs baseline {base_mean:.2f} ± {base_se:.2f} (Cohen\\'s d={d:.1f}, p_FDR<{p_fdr:.1e}), representing a {ratio:+.0f}% change.\"')

    print("\n" + "=" * 80)
    print("✅ Track F Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Track F results")
    parser.add_argument("--input", type=str, required=True, help="Path to track_f_episode_metrics.csv")
    parser.add_argument("--output", type=str, default="logs/track_f", help="Output directory")

    args = parser.parse_args()

    analyze_track_f(Path(args.input), Path(args.output))
```

---

## 🚀 Part 3: Execution Plan

### Step 1: Apply Patches
```bash
cd /srv/luminous-dynamics/kosmic-lab

# Backup current runner
cp fre/track_f_runner.py fre/track_f_runner.py.backup

# Apply patches (manual edit OR use script below)
# ... apply all 6 patches listed above ...
```

### Step 2: Create Analysis Script
```bash
# Copy analysis script from Part 2
# Save as fre/analyze_track_f.py
chmod +x fre/analyze_track_f.py
```

### Step 3: Re-run Track F
```bash
source .venv/bin/activate

# Clear old logs
rm -f logs/track_f/*.csv logs/track_f/*.npz

# Re-run with corrected FGSM
python3 fre/track_f_runner.py --config fre/configs/track_f_adversarial.yaml

# Should take ~30-45 minutes for 150 episodes
```

### Step 4: Analyze Results
```bash
# Generate statistics
python3 fre/analyze_track_f.py \
    --input logs/track_f/track_f_episode_metrics.csv \
    --output logs/track_f

# Outputs:
# - track_f_summary.csv (mean ± SE, CI for each condition)
# - track_f_comparisons.csv (Cohen's d, p-values, FDR)
# - Manuscript text snippets (printed to console)
```

### Step 5: Update Documentation
```bash
# Update TRACK_F_ADVERSARIAL_RESULTS.md with new numbers
# Update CROSS_TRACK_ANALYSIS.md if ratios changed
# Update PAPER_5_UNIFIED_THEORY_OUTLINE.md with corrected text
```

---

## ✅ Expected Outcomes

### If FGSM Enhancement Holds (~+85%)
- Adversarial K = 1.10-1.20 (vs 0.60-0.65 baseline)
- Cohen's d > 2.0
- p_FDR < 0.001
- >95% FGSM steps increase loss

### If Effect Attenuated but Significant (~+20-40%)
- Adversarial K = 0.75-0.90 (vs 0.60-0.65 baseline)
- Cohen's d > 1.0
- p_FDR < 0.01
- >90% FGSM steps increase loss

### If Not Significant (Fallback)
- Document that FGSM correction changed results
- Emphasize other Track F findings (reward spoofing, noise)
- Focus paper on Tracks B-E findings + developmental/topology

---

## 📋 Verification Checklist

- [ ] All 6 patches applied to track_f_runner.py
- [ ] analyze_track_f.py script created and executable
- [ ] Track F re-run completed (150 episodes)
- [ ] track_f_episode_metrics.csv generated
- [ ] fgsm_sanity_checks.csv generated
- [ ] Analysis script produces summary & comparisons tables
- [ ] FGSM sanity check shows >90% loss increase
- [ ] Manuscript text snippets generated
- [ ] TRACK_F_ADVERSARIAL_RESULTS.md updated
- [ ] CROSS_TRACK_ANALYSIS.md updated
- [ ] PAPER_5_UNIFIED_THEORY_OUTLINE.md updated

---

**Ready to proceed?** Apply patches → Re-run Track F → Analyze → Update paper! 🚀
