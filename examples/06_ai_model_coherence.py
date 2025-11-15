#!/usr/bin/env python3
"""
🤖 Example 6: AI Model Internal Coherence Analysis

This example demonstrates using K-Index to analyze internal coherence
in neural networks. We examine whether a model's internal representations
align with its predictions and the true labels.

**Research Application**: Interpretable AI, model debugging, alignment research,
consciousness in artificial systems, prediction reliability assessment.

**Use Case**: Analyzing whether an AI model's "understanding" (internal representations)
is coherent with what it predicts, useful for:
- Detecting when models are uncertain but confident (overconfidence)
- Understanding representation quality in different layers
- Validating alignment between model internals and outputs

Author: Kosmic Lab Contributors
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import bootstrap_confidence_interval, infer_git_sha
from core.logging_config import setup_logging, get_logger
from core.kcodex import KCodexWriter
from fre.metrics.k_index import k_index, bootstrap_k_ci

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)


class SimpleMLP:
    """
    Simple Multi-Layer Perceptron for demonstration.

    In practice, replace with PyTorch/TensorFlow models and extract
    real hidden layer activations.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1, seed: int = 42):
        """Initialize MLP with random weights."""
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights (He initialization)
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        logger.info(f"MLP initialized: {input_dim} → {hidden_dim} → {output_dim}")

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X: np.ndarray, return_hidden: bool = False) -> Tuple:
        """
        Forward pass.

        Args:
            X: Input data (n_samples, input_dim)
            return_hidden: If True, return hidden activations

        Returns:
            Predictions and optionally hidden activations
        """
        # Layer 1
        z1 = X @ self.W1 + self.b1
        h1 = self.relu(z1)

        # Layer 2
        z2 = h1 @ self.W2 + self.b2
        output = self.sigmoid(z2).flatten()

        if return_hidden:
            return output, h1
        return output

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01):
        """
        Single training step (simplified SGD).

        Args:
            X: Input data
            y: Target labels
            lr: Learning rate
        """
        n = len(X)

        # Forward pass
        z1 = X @ self.W1 + self.b1
        h1 = self.relu(z1)
        z2 = h1 @ self.W2 + self.b2
        output = self.sigmoid(z2).flatten()

        # Backward pass (simplified)
        loss = np.mean((output - y) ** 2)

        # Gradients
        d_output = 2 * (output - y) / n
        d_z2 = d_output * output * (1 - output)  # sigmoid derivative
        d_W2 = h1.T @ d_z2.reshape(-1, 1)
        d_b2 = np.sum(d_z2)

        d_h1 = d_z2.reshape(-1, 1) @ self.W2.T
        d_z1 = d_h1 * (z1 > 0)  # ReLU derivative
        d_W1 = X.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        # Update weights
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

        return loss


class AICoherenceAnalyzer:
    """Analyzes internal coherence of AI models using K-Index."""

    def __init__(self, seed: int = 42):
        """Initialize analyzer."""
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        logger.info("AI Coherence Analyzer initialized")

    def generate_synthetic_dataset(
        self,
        n_samples: int = 1000,
        input_dim: int = 10,
        difficulty: str = "medium"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification dataset.

        Args:
            n_samples: Number of samples
            input_dim: Input dimensionality
            difficulty: Dataset difficulty ("easy", "medium", "hard")

        Returns:
            X (features), y (labels)
        """
        # Generate features
        X = self.rng.normal(0, 1, (n_samples, input_dim))

        # Generate labels based on a linear combination + nonlinearity
        if difficulty == "easy":
            weights = self.rng.normal(0, 1, input_dim)
            linear_combination = X @ weights
            y = (linear_combination > 0).astype(float)

        elif difficulty == "medium":
            weights = self.rng.normal(0, 1, input_dim)
            linear_combination = X @ weights
            y = (np.tanh(linear_combination) > 0).astype(float)

        else:  # hard
            # XOR-like problem
            weights1 = self.rng.normal(0, 1, input_dim)
            weights2 = self.rng.normal(0, 1, input_dim)
            term1 = X @ weights1
            term2 = X @ weights2
            y = ((term1 > 0) != (term2 > 0)).astype(float)

        logger.info(f"Generated {difficulty} dataset: {X.shape}, balance={y.mean():.2f}")

        return X, y

    def analyze_representation_coherence(
        self,
        hidden_representations: np.ndarray,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Dict:
        """
        Analyze coherence between hidden representations and predictions/labels.

        We use the L2 norm of hidden representations as a proxy for model "confidence"
        or "internal certainty". High coherence means the model's internal state
        aligns well with its predictions.

        Args:
            hidden_representations: Hidden layer activations (n_samples, hidden_dim)
            predictions: Model predictions (n_samples,)
            true_labels: Ground truth labels (n_samples,)
            n_bootstrap: Bootstrap iterations

        Returns:
            Dictionary with coherence metrics
        """
        logger.info("Analyzing representation coherence...")

        # Compute representation magnitude (proxy for model certainty)
        repr_magnitude = np.linalg.norm(hidden_representations, axis=1)
        repr_magnitude = (repr_magnitude - repr_magnitude.min()) / (repr_magnitude.max() - repr_magnitude.min())

        # Coherence 1: Internal representation vs predictions
        k_repr_pred, ci_low_rp, ci_high_rp = bootstrap_k_ci(
            repr_magnitude,
            predictions,
            n_bootstrap=n_bootstrap,
            seed=self.seed
        )

        # Coherence 2: Predictions vs true labels
        k_pred_true, ci_low_pt, ci_high_pt = bootstrap_k_ci(
            predictions,
            true_labels,
            n_bootstrap=n_bootstrap,
            seed=self.seed
        )

        # Coherence 3: Internal representation vs true labels
        k_repr_true, ci_low_rt, ci_high_rt = bootstrap_k_ci(
            repr_magnitude,
            true_labels,
            n_bootstrap=n_bootstrap,
            seed=self.seed
        )

        # Compute prediction accuracy
        accuracy = np.mean((predictions > 0.5) == true_labels)

        # Detect overconfidence: high repr magnitude but wrong predictions
        wrong_predictions = (predictions > 0.5) != true_labels
        overconfidence_score = np.mean(repr_magnitude[wrong_predictions]) if np.any(wrong_predictions) else 0.0

        results = {
            "k_repr_pred": k_repr_pred,
            "k_repr_pred_ci": (ci_low_rp, ci_high_rp),
            "k_pred_true": k_pred_true,
            "k_pred_true_ci": (ci_low_pt, ci_high_pt),
            "k_repr_true": k_repr_true,
            "k_repr_true_ci": (ci_low_rt, ci_high_rt),
            "accuracy": accuracy,
            "overconfidence_score": overconfidence_score,
            "repr_magnitude_mean": np.mean(repr_magnitude),
            "repr_magnitude_std": np.std(repr_magnitude)
        }

        logger.info(f"K(repr, pred): {k_repr_pred:.4f}")
        logger.info(f"K(pred, true): {k_pred_true:.4f}")
        logger.info(f"K(repr, true): {k_repr_true:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")

        return results


def visualize_ai_coherence(
    X: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    hidden: np.ndarray,
    coherence: Dict
):
    """Visualize AI model coherence analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Compute representation magnitude
    repr_mag = np.linalg.norm(hidden, axis=1)
    repr_mag_norm = (repr_mag - repr_mag.min()) / (repr_mag.max() - repr_mag.min())

    # Plot 1: Predictions vs True Labels
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y, predictions, alpha=0.5, s=20, c=repr_mag_norm, cmap='viridis')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel("True Labels", fontsize=12)
    ax1.set_ylabel("Predictions", fontsize=12)
    ax1.set_title(f"Predictions (Acc={coherence['accuracy']:.3f})", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Representation Magnitude Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    correct = (predictions > 0.5) == y
    ax2.hist(repr_mag_norm[correct], bins=30, alpha=0.7, label='Correct', color='green')
    ax2.hist(repr_mag_norm[~correct], bins=30, alpha=0.7, label='Wrong', color='red')
    ax2.set_xlabel("Normalized Representation Magnitude", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Internal Certainty Distribution", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Coherence Metrics Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['K(repr,pred)', 'K(pred,true)', 'K(repr,true)']
    values = [coherence['k_repr_pred'], coherence['k_pred_true'], coherence['k_repr_true']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel("K-Index", fontsize=12)
    ax3.set_title("Coherence Metrics", fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Hidden Representations (PCA to 2D)
    ax4 = fig.add_subplot(gs[1, :2])
    # Simple PCA
    hidden_centered = hidden - hidden.mean(axis=0)
    cov = hidden_centered.T @ hidden_centered / len(hidden)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pca_components = eigenvectors[:, -2:]  # Top 2 components
    hidden_2d = hidden_centered @ pca_components

    scatter = ax4.scatter(hidden_2d[:, 0], hidden_2d[:, 1],
                         c=predictions, cmap='RdYlGn', s=30, alpha=0.6,
                         edgecolors='black', linewidth=0.5)
    ax4.set_xlabel("PC1", fontsize=12)
    ax4.set_ylabel("PC2", fontsize=12)
    ax4.set_title("Hidden Representations (colored by predictions)", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Prediction')
    ax4.grid(alpha=0.3)

    # Plot 5: Confidence Calibration
    ax5 = fig.add_subplot(gs[1, 2])
    # Bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    for i in range(len(bins) - 1):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if np.any(mask):
            bin_accs.append(np.mean((predictions[mask] > 0.5) == y[mask]))
        else:
            bin_accs.append(np.nan)

    ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax5.plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8, label='Model')
    ax5.set_xlabel("Predicted Probability", fontsize=12)
    ax5.set_ylabel("Empirical Accuracy", fontsize=12)
    ax5.set_title("Calibration Curve", fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])

    # Plot 6: Summary Statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    summary = f"""
    🤖 AI MODEL COHERENCE ANALYSIS SUMMARY

    Model Performance:
      • Accuracy: {coherence['accuracy']:.4f} ({int(coherence['accuracy']*100)}% correct)
      • Overconfidence Score: {coherence['overconfidence_score']:.4f} (lower is better)

    Internal Coherence (K-Index):
      • K(representation, prediction): {coherence['k_repr_pred']:.4f} [{coherence['k_repr_pred_ci'][0]:.3f}, {coherence['k_repr_pred_ci'][1]:.3f}]
          → How well internal "understanding" aligns with predictions

      • K(prediction, true): {coherence['k_pred_true']:.4f} [{coherence['k_pred_true_ci'][0]:.3f}, {coherence['k_pred_true_ci'][1]:.3f}]
          → How well predictions match reality

      • K(representation, true): {coherence['k_repr_true']:.4f} [{coherence['k_repr_true_ci'][0]:.3f}, {coherence['k_repr_true_ci'][1]:.3f}]
          → How well internal "understanding" matches reality

    Representation Statistics:
      • Mean magnitude: {coherence['repr_magnitude_mean']:.4f}
      • Std magnitude: {coherence['repr_magnitude_std']:.4f}

    Interpretation:
      • Model Alignment: {"✓ Excellent" if coherence['k_repr_pred'] > 0.8 else "✓ Good" if coherence['k_repr_pred'] > 0.6 else "⚠ Moderate" if coherence['k_repr_pred'] > 0.4 else "✗ Poor"}
      • Prediction Quality: {"✓ Excellent" if coherence['k_pred_true'] > 0.8 else "✓ Good" if coherence['k_pred_true'] > 0.6 else "⚠ Moderate" if coherence['k_pred_true'] > 0.4 else "✗ Poor"}
      • Internal Understanding: {"✓ Excellent" if coherence['k_repr_true'] > 0.8 else "✓ Good" if coherence['k_repr_true'] > 0.6 else "⚠ Moderate" if coherence['k_repr_true'] > 0.4 else "✗ Poor"}
    """

    ax6.text(0.05, 0.5, summary, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.suptitle("🤖 AI Model Internal Coherence Analysis with K-Index",
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    """Run complete AI coherence analysis."""
    logger.info("=" * 80)
    logger.info("🤖 Example 6: AI Model Internal Coherence Analysis")
    logger.info("=" * 80)

    # Create K-Codex writer
    kcodex_path = Path("logs/example_06_ai_coherence_kcodex.json")
    kcodex_path.parent.mkdir(parents=True, exist_ok=True)
    kcodex = KCodexWriter(str(kcodex_path))

    # Initialize analyzer
    analyzer = AICoherenceAnalyzer(seed=SEED)

    # Generate dataset
    n_samples = 2000
    input_dim = 10
    difficulty = "medium"

    logger.info(f"\n--- Generating {difficulty} dataset ---")
    X, y = analyzer.generate_synthetic_dataset(n_samples, input_dim, difficulty)

    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    logger.info("\n--- Training model ---")
    model = SimpleMLP(input_dim=input_dim, hidden_dim=64, output_dim=1, seed=SEED)

    n_epochs = 100
    for epoch in range(n_epochs):
        loss = model.train_step(X_train, y_train, lr=0.1)
        if (epoch + 1) % 20 == 0:
            preds_train = model.forward(X_train)
            acc_train = np.mean((preds_train > 0.5) == y_train)
            logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss={loss:.4f}, Train Acc={acc_train:.4f}")

    # Evaluate on test set
    logger.info("\n--- Evaluating model ---")
    predictions, hidden_representations = model.forward(X_test, return_hidden=True)

    # Analyze coherence
    logger.info("\n--- Analyzing coherence ---")
    coherence = analyzer.analyze_representation_coherence(
        hidden_representations,
        predictions,
        y_test,
        n_bootstrap=1000
    )

    # Log to K-Codex
    kcodex.log_experiment(
        experiment_name="ai_model_coherence_analysis",
        params={
            "n_samples": n_samples,
            "input_dim": input_dim,
            "hidden_dim": 64,
            "difficulty": difficulty,
            "n_epochs": n_epochs,
            "learning_rate": 0.1,
            "train_test_split": 0.8
        },
        metrics={
            "accuracy": coherence["accuracy"],
            "k_repr_pred": coherence["k_repr_pred"],
            "k_pred_true": coherence["k_pred_true"],
            "k_repr_true": coherence["k_repr_true"],
            "overconfidence_score": coherence["overconfidence_score"]
        },
        seed=SEED,
        extra_metadata={
            "example": "06_ai_model_coherence",
            "application": "interpretable_ai",
            "model_type": "mlp"
        }
    )

    # Visualize
    logger.info("\n📊 Generating visualization...")
    fig = visualize_ai_coherence(X_test, y_test, predictions, hidden_representations, coherence)

    # Save
    output_path = Path("outputs/example_06_ai_coherence.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Visualization saved: {output_path}")

    plt.show()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📈 ANALYSIS COMPLETE")
    logger.info("=" * 80)
    print(f"\n✅ Accuracy: {coherence['accuracy']:.4f}")
    print(f"✅ K(repr, pred): {coherence['k_repr_pred']:.4f} - Internal alignment")
    print(f"✅ K(pred, true): {coherence['k_pred_true']:.4f} - Prediction quality")
    print(f"✅ K(repr, true): {coherence['k_repr_true']:.4f} - Understanding quality")
    print(f"\n💡 Insight: {'Model internals are well-aligned!' if coherence['k_repr_pred'] > 0.7 else 'Model may have internal misalignment.'}")

    logger.info(f"\n✅ K-Codex saved: {kcodex_path}")
    logger.info(f"✅ Git SHA: {infer_git_sha()}")
    logger.info("\n🎓 Example complete! See outputs/example_06_ai_coherence.png")


if __name__ == "__main__":
    main()
