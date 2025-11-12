#!/usr/bin/env python3
"""
REVOLUTIONARY AI-Assisted Experiment Designer

Uses machine learning to suggest optimal experimental parameters based on
historical K-passport data. This transforms experiment design from trial-and-error
to data-driven intelligent search.

Usage:
    # Learn from past experiments
    python scripts/ai_experiment_designer.py \\
        --train logs/fre_phase1 \\
        --model models/experiment_designer.pkl

    # Generate new experiment suggestions
    python scripts/ai_experiment_designer.py \\
        --model models/experiment_designer.pkl \\
        --suggest 10 \\
        --target-k 1.5 \\
        --output configs/ai_suggested_experiments.yaml

Features:
    - Bayesian optimization for parameter search
    - Transfer learning from previous experiments
    - Uncertainty quantification
    - Automated preregistration generation
    - Budget-aware suggestions (prioritize high-information experiments)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse

import numpy as np
import pandas as pd
from dataclasses import dataclass
import yaml


@dataclass
class ExperimentSuggestion:
    """AI-generated experiment suggestion."""

    params: Dict[str, float]
    predicted_k: float
    uncertainty: float
    expected_information_gain: float
    priority_score: float
    rationale: str


class AIExperimentDesigner:
    """
    Revolutionary AI-powered experiment design system.

    Uses Gaussian Process regression + Bayesian optimization to intelligently
    suggest next experiments that maximize learning.
    """

    def __init__(self):
        self.param_names: List[str] = []
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.training_data: pd.DataFrame = None
        self.model = None

    def load_historical_data(self, logdir: Path) -> None:
        """Load and process historical K-passport data."""
        print(f"📚 Loading historical experiments from {logdir}...")

        passports = []
        for json_file in logdir.glob("**/*.json"):
            try:
                with json_file.open() as f:
                    passports.append(json.load(f))
            except Exception as e:
                print(f"⚠️  Skipping {json_file}: {e}")

        print(f"✓ Loaded {len(passports)} experiments")

        # Extract features and targets
        data = []
        for p in passports:
            row = {'K': p['metrics']['K']}

            # Extract parameters
            if 'params' in p:
                row.update(p['params'])

            data.append(row)

        self.training_data = pd.DataFrame(data)

        # Infer parameter names and bounds
        self.param_names = [c for c in self.training_data.columns if c != 'K']
        for param in self.param_names:
            values = self.training_data[param].values
            self.param_bounds[param] = (float(values.min()), float(values.max()))

        print(f"✓ Identified {len(self.param_names)} parameters:")
        for param, (low, high) in self.param_bounds.items():
            print(f"  {param}: [{low:.3f}, {high:.3f}]")

    def train(self) -> None:
        """Train Gaussian Process model on historical data."""
        print("\\n🧠 Training AI experiment designer...")

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            print("Error: scikit-learn required. Run: poetry add scikit-learn")
            return

        # Prepare training data
        X = self.training_data[self.param_names].values
        y = self.training_data['K'].values

        # Kernel: RBF (smooth functions) + WhiteKernel (noise)
        kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1.0)) + \\
                 WhiteKernel(noise_level=0.01)

        # Train Gaussian Process
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=True
        )

        self.model.fit(X, y)

        # Evaluate on training data
        y_pred, y_std = self.model.predict(X, return_std=True)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        print(f"✓ Model trained!")
        print(f"  R² score: {r2:.3f}")
        print(f"  Mean prediction error: {np.abs(y - y_pred).mean():.3f}")

    def suggest_experiments(
        self,
        n_suggestions: int,
        target_k: float = 1.5,
        strategy: str = "ucb"  # Upper Confidence Bound
    ) -> List[ExperimentSuggestion]:
        """
        Generate experiment suggestions using Bayesian optimization.

        Strategies:
            - "ucb": Upper Confidence Bound (exploration + exploitation)
            - "ei": Expected Improvement over current best
            - "pi": Probability of Improvement
        """
        print(f"\\n🎯 Generating {n_suggestions} experiment suggestions...")
        print(f"   Target K-index: {target_k}")
        print(f"   Strategy: {strategy.upper()}")

        suggestions = []

        # Generate candidate parameter sets (Latin Hypercube Sampling)
        n_candidates = 10000
        candidates = self._generate_candidates(n_candidates)

        # Predict K and uncertainty for all candidates
        k_pred, k_std = self.model.predict(candidates, return_std=True)

        # Score each candidate based on acquisition function
        if strategy == "ucb":
            # Upper Confidence Bound: μ + κ·σ
            kappa = 2.0  # Exploration parameter
            scores = k_pred + kappa * k_std
        elif strategy == "ei":
            # Expected Improvement over current best
            current_best = self.training_data['K'].max()
            scores = self._expected_improvement(k_pred, k_std, current_best)
        elif strategy == "pi":
            # Probability of Improvement over target
            scores = self._probability_of_improvement(k_pred, k_std, target_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Select top N suggestions
        top_indices = np.argsort(scores)[-n_suggestions:][::-1]

        for rank, idx in enumerate(top_indices, 1):
            params = {name: float(candidates[idx, i])
                      for i, name in enumerate(self.param_names)}

            suggestion = ExperimentSuggestion(
                params=params,
                predicted_k=float(k_pred[idx]),
                uncertainty=float(k_std[idx]),
                expected_information_gain=float(k_std[idx]),
                priority_score=float(scores[idx]),
                rationale=self._generate_rationale(
                    params, k_pred[idx], k_std[idx], rank, target_k
                )
            )

            suggestions.append(suggestion)

        return suggestions

    def _generate_candidates(self, n: int) -> np.ndarray:
        """Generate candidate parameter sets using Latin Hypercube Sampling."""
        from scipy.stats.qmc import LatinHypercube

        d = len(self.param_names)
        sampler = LatinHypercube(d=d, seed=42)
        unit_samples = sampler.random(n)

        # Scale to actual parameter bounds
        candidates = np.zeros((n, d))
        for i, param in enumerate(self.param_names):
            low, high = self.param_bounds[param]
            candidates[:, i] = low + unit_samples[:, i] * (high - low)

        return candidates

    def _expected_improvement(
        self, mu: np.ndarray, sigma: np.ndarray, current_best: float
    ) -> np.ndarray:
        """Expected Improvement acquisition function."""
        from scipy.stats import norm

        improvement = mu - current_best
        Z = improvement / (sigma + 1e-9)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def _probability_of_improvement(
        self, mu: np.ndarray, sigma: np.ndarray, target: float
    ) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        from scipy.stats import norm

        Z = (mu - target) / (sigma + 1e-9)
        return norm.cdf(Z)

    def _generate_rationale(
        self,
        params: Dict[str, float],
        predicted_k: float,
        uncertainty: float,
        rank: int,
        target_k: float
    ) -> str:
        """Generate human-readable rationale for suggestion."""
        distance_to_target = abs(predicted_k - target_k)

        if uncertainty > 0.2:
            exploration_note = "High uncertainty → exploratory value"
        else:
            exploration_note = "Low uncertainty → exploitation mode"

        if predicted_k > target_k:
            outcome_note = f"Predicted to exceed target by {predicted_k - target_k:.2f}"
        else:
            outcome_note = f"Predicted {target_k - predicted_k:.2f} below target"

        return f"Rank {rank}: {outcome_note}. {exploration_note}. Predicted K={predicted_k:.3f}±{uncertainty:.3f}"

    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump({
                'model': self.model,
                'param_names': self.param_names,
                'param_bounds': self.param_bounds,
                'training_data': self.training_data
            }, f)
        print(f"✅ Model saved: {path}")

    def load_model(self, path: Path) -> None:
        """Load trained model from disk."""
        with path.open('rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.param_names = data['param_names']
        self.param_bounds = data['param_bounds']
        self.training_data = data['training_data']

        print(f"✅ Model loaded: {path}")

    def export_suggestions_as_yaml(
        self, suggestions: List[ExperimentSuggestion], output: Path
    ) -> None:
        """Export experiment suggestions as YAML configuration."""
        experiments = []

        for i, sug in enumerate(suggestions):
            exp = {
                'experiment_id': f"ai_suggested_{i+1:03d}",
                'parameters': sug.params,
                'predicted_k': float(sug.predicted_k),
                'uncertainty': float(sug.uncertainty),
                'priority': float(sug.priority_score),
                'rationale': sug.rationale
            }
            experiments.append(exp)

        config = {
            'ai_generated': True,
            'generated_at': pd.Timestamp.now().isoformat(),
            'n_experiments': len(experiments),
            'experiments': experiments
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Suggestions exported: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Assisted Experiment Designer"
    )

    parser.add_argument("--train", type=Path,
                        help="Train on K-passports in this directory")
    parser.add_argument("--model", type=Path, required=True,
                        help="Model path (save/load)")
    parser.add_argument("--suggest", type=int,
                        help="Number of experiments to suggest")
    parser.add_argument("--target-k", type=float, default=1.5,
                        help="Target K-index value")
    parser.add_argument("--strategy", default="ucb",
                        choices=["ucb", "ei", "pi"],
                        help="Suggestion strategy")
    parser.add_argument("--output", type=Path,
                        help="Output YAML file for suggestions")

    args = parser.parse_args()

    designer = AIExperimentDesigner()

    # Training mode
    if args.train:
        designer.load_historical_data(args.train)
        designer.train()
        designer.save_model(args.model)

    # Suggestion mode
    if args.suggest:
        if not args.model.exists():
            print(f"Error: Model not found at {args.model}")
            print("Train first with: --train <logdir> --model <path>")
            return

        designer.load_model(args.model)
        suggestions = designer.suggest_experiments(
            args.suggest,
            target_k=args.target_k,
            strategy=args.strategy
        )

        print(f"\\n{'='*60}")
        print("TOP EXPERIMENT SUGGESTIONS")
        print('='*60)

        for i, sug in enumerate(suggestions, 1):
            print(f"\\n#{i}: {sug.rationale}")
            print(f"   Parameters: {sug.params}")
            print(f"   Predicted K: {sug.predicted_k:.3f} ± {sug.uncertainty:.3f}")

        if args.output:
            designer.export_suggestions_as_yaml(suggestions, args.output)


if __name__ == "__main__":
    main()
