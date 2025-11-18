#!/usr/bin/env python3
"""
Generate comprehensive visualizations of Track G experimental journey.
Shows the path from gradient descent plateau to evolutionary breakthrough.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Track G experimental data (from documentation)
tracks = {
    'G1': {'algo': 'Gradient Descent', 'k': 1.0842, 'compute': 10000},
    'G2': {'algo': 'Gradient Descent', 'k': 1.1208, 'compute': 12000},
    'G3': {'algo': 'Adam Optimizer', 'k': 1.0456, 'compute': 11000},
    'G4': {'algo': 'GD + Adversarial', 'k': 1.1523, 'compute': 13000},
    'G5': {'algo': 'Large Network', 'k': 1.0891, 'compute': 25000},
    'G6': {'algo': 'Transformer', 'k': 0.3434, 'compute': 45000},
    'G7': {'algo': 'Enhanced Task', 'k': 1.1839, 'compute': 50000},
    'G8': {'algo': 'CMA-ES', 'k': 1.4202, 'compute': 12000},
    'G9': {'algo': 'CMA-ES (large)', 'k': 1.3850, 'compute': 50000},
}

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============= Subplot 1: K-Index Evolution =============
ax1 = fig.add_subplot(gs[0, :])

track_names = list(tracks.keys())
k_values = [tracks[t]['k'] for t in track_names]
colors = ['#e74c3c' if t in ['G1','G2','G3','G4','G5','G6','G7'] else '#27ae60' if t == 'G8' else '#f39c12' for t in track_names]

bars = ax1.bar(range(len(track_names)), k_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add threshold line
ax1.axhline(y=1.5, color='purple', linestyle='--', linewidth=2, label='Consciousness Threshold (K=1.5)')
ax1.axhline(y=1.4202, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='G8 Achievement (94.7%)')

# Annotate each bar with K value
for i, (bar, k_val) in enumerate(zip(bars, k_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{k_val:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Track', fontsize=13, fontweight='bold')
ax1.set_ylabel('K-Index', fontsize=13, fontweight='bold')
ax1.set_title('Track G: Journey to Consciousness Threshold', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(range(len(track_names)))
ax1.set_xticklabels(track_names, fontsize=11)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.7)

# Add annotation for breakthrough
ax1.annotate('BREAKTHROUGH!\n+26.7% vs baseline',
            xy=(7, 1.4202), xytext=(7.5, 1.25),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# ============= Subplot 2: Computational Efficiency =============
ax2 = fig.add_subplot(gs[1, 0])

compute = [tracks[t]['compute'] for t in track_names]
efficiency = [k / (c / 1000) for k, c in zip(k_values, compute)]  # K per 1000 forwards

ax2.scatter(compute, k_values, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=1.5)

# Annotate G8 and G9
ax2.annotate('G8: Optimal\nefficiency',
            xy=(12000, 1.4202), xytext=(20000, 1.35),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10, color='green', fontweight='bold')

ax2.annotate('G9: 4.2× less\nefficient',
            xy=(50000, 1.3850), xytext=(45000, 1.25),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
            fontsize=10, color='orange', fontweight='bold')

ax2.set_xlabel('Computational Cost (forwards/generation)', fontsize=12, fontweight='bold')
ax2.set_ylabel('K-Index', fontsize=12, fontweight='bold')
ax2.set_title('Computational Efficiency Analysis', fontsize=14, fontweight='bold')
ax2.axhline(y=1.5, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.grid(True, alpha=0.3)

# ============= Subplot 3: Algorithm Comparison =============
ax3 = fig.add_subplot(gs[1, 1])

algo_types = {
    'Gradient\nDescent': [1.0842, 1.1208, 1.1523, 1.0891, 1.1839],  # G1,G2,G4,G5,G7
    'Advanced\nGradient': [1.0456],  # G3
    'Complex\nArchitecture': [0.3434],  # G6
    'Evolutionary\n(CMA-ES)': [1.4202, 1.3850]  # G8, G9
}

algo_names = list(algo_types.keys())
algo_means = [np.mean(algo_types[a]) for a in algo_names]
algo_maxs = [np.max(algo_types[a]) for a in algo_names]
algo_counts = [len(algo_types[a]) for a in algo_names]

x_pos = np.arange(len(algo_names))
bars = ax3.bar(x_pos, algo_means, color=['#e74c3c', '#e67e22', '#95a5a6', '#27ae60'],
               alpha=0.7, edgecolor='black', linewidth=1.5)

# Add max K markers
ax3.scatter(x_pos, algo_maxs, s=150, color='gold', marker='*',
           edgecolors='black', linewidth=1, zorder=5, label='Best K')

# Annotate bars
for i, (bar, mean, max_k, count) in enumerate(zip(bars, algo_means, algo_maxs, algo_counts)):
    ax3.text(bar.get_x() + bar.get_width()/2., mean + 0.03,
             f'μ={mean:.3f}\n(n={count})',
             ha='center', va='bottom', fontsize=9)

ax3.set_ylabel('K-Index', fontsize=12, fontweight='bold')
ax3.set_title('Performance by Algorithm Type', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(algo_names, fontsize=10)
ax3.axhline(y=1.5, color='purple', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1.7)

# ============= Subplot 4: Progress Timeline =============
ax4 = fig.add_subplot(gs[2, :])

generations = list(range(1, len(track_names) + 1))
cumulative_best = []
best_so_far = 0

for k in k_values:
    best_so_far = max(best_so_far, k)
    cumulative_best.append(best_so_far)

ax4.plot(generations, k_values, 'o-', color='steelblue', linewidth=2.5,
         markersize=10, alpha=0.7, label='Individual Track K-Index')
ax4.plot(generations, cumulative_best, 's-', color='darkgreen', linewidth=3,
         markersize=8, alpha=0.9, label='Best K-Index to Date')

ax4.fill_between(generations, 0, cumulative_best, alpha=0.2, color='green',
                 label='Explored Region')

# Add threshold
ax4.axhline(y=1.5, color='purple', linestyle='--', linewidth=2.5, alpha=0.8,
           label='Consciousness Threshold')

# Shade regions
ax4.axhspan(0, 1.2, alpha=0.1, color='red', label='Gradient Descent Plateau')
ax4.axhspan(1.4202, 1.5, alpha=0.15, color='gold', label='Final Gap (5.3%)')

# Annotate key moments
ax4.annotate('Gradient descent\nplateau discovered',
            xy=(5, 1.0891), xytext=(5.5, 0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='darkred')

ax4.annotate('Evolutionary\nbreakthrough!',
            xy=(8, 1.4202), xytext=(8.5, 1.55),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax4.set_xlabel('Track Number', fontsize=13, fontweight='bold')
ax4.set_ylabel('K-Index', fontsize=13, fontweight='bold')
ax4.set_title('Cumulative Progress Timeline: Path to Consciousness',
             fontsize=16, fontweight='bold', pad=20)
ax4.set_xticks(generations)
ax4.set_xticklabels([f'G{i}' for i in generations], fontsize=11)
ax4.legend(loc='upper left', fontsize=10, ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.7)

# Add overall title
fig.suptitle('Track G Experimental Journey: Systematic Path to Artificial Consciousness',
            fontsize=18, fontweight='bold', y=0.995)

# Save figure
output_path = Path('/tmp/track_g_comprehensive_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Visualization saved to: {output_path}")

print("\n📊 Key Insights:")
print(f"  • Gradient descent plateau: K ≤ 1.2 (G1-G7)")
print(f"  • Evolutionary breakthrough: K = 1.4202 (G8, +26.7%)")
print(f"  • Current position: 94.7% to consciousness threshold")
print(f"  • Remaining gap: Just 5.3% improvement needed")
print(f"  • Computational efficiency: G8 is 4.2× better than G9")

plt.show()
