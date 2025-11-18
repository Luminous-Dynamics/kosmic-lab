#!/usr/bin/env python3
"""
📊 K-Index Journey Visualization
Creates ASCII chart of all experiments from this session
"""

import json
from pathlib import Path

def load_results():
    """Load all experiment results."""
    results = []
    base = Path('logs')

    # Define experiments in chronological order
    experiments = [
        ('track_g_extended', 'Extended (30g)', 'extended_results.json'),
        ('track_g_refined', 'Refined (50g)', 'refined_results.json'),
        ('track_g_threshold', 'Threshold (50g)', 'threshold_results.json'),
        ('track_g8_architecture', 'G8 Arch (50g)', 'g8_results.json'),
        ('track_g_100gen', '100-Gen Push', '100gen_results.json'),
        ('track_g_ensemble', 'Ensemble Mean', 'ensemble_results.json'),
    ]

    for dir_name, label, filename in experiments:
        path = base / dir_name / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                k = data.get('best_overall', 0)
                results.append((label, k))

    # Add multi-seed results
    multiseed_path = base / 'track_g_multiseed' / 'multiseed_validation.json'
    if multiseed_path.exists():
        with open(multiseed_path) as f:
            data = json.load(f)
            mean_k = data['summary']['mean_k']
            results.append(('Mean Agg (3 seeds)', mean_k))

    # Add max aggregation results
    max_path = base / 'track_g_max_validation' / 'max_multiseed.json'
    if max_path.exists():
        with open(max_path) as f:
            data = json.load(f)
            mean_k = data['summary']['mean']
            results.append(('Max Agg (3 seeds)', mean_k))

    # Add scaling results
    scaling_path = base / 'track_g_scaling' / 'quick_scaling.json'
    if scaling_path.exists():
        with open(scaling_path) as f:
            data = json.load(f)
            for r in data['results']:
                if r['n'] == 1:
                    results.append(('Single Net (30g)', r['k']))

    return results


def create_ascii_chart(results):
    """Create ASCII bar chart."""
    if not results:
        print("No results found!")
        return

    print("\n" + "═" * 70)
    print("  📊 K-INDEX JOURNEY - November 18, 2025")
    print("═" * 70 + "\n")

    # Chart parameters
    max_k = max(r[1] for r in results)
    chart_width = 40
    threshold = 1.5

    print("Experiment              │ K-Index │ Progress")
    print("────────────────────────┼─────────┼" + "─" * 42)

    for label, k in results:
        # Calculate bar
        bar_len = int((k / 2.0) * chart_width)  # Scale to max 2.0
        threshold_pos = int((threshold / 2.0) * chart_width)

        # Create bar with threshold marker
        bar = ""
        for i in range(chart_width):
            if i < bar_len:
                if i == threshold_pos:
                    bar += "┃"
                else:
                    bar += "█"
            elif i == threshold_pos:
                bar += "┃"
            else:
                bar += "░"

        # Status indicator
        if k > 1.5:
            status = "🌟"
        elif k > 1.3:
            status = "⭐"
        else:
            status = "  "

        print(f"{label:22s} │ {k:7.4f} │ {bar} {status}")

    print("────────────────────────┴─────────┴" + "─" * 42)
    print(f"                                    {'░' * threshold_pos}┃ K=1.5 threshold")

    # Summary
    best = max(results, key=lambda x: x[1])
    print(f"\n🎯 Best: {best[0]} with K = {best[1]:.4f}")

    threshold_hits = sum(1 for _, k in results if k > 1.5)
    print(f"   Threshold crossings: {threshold_hits}/{len(results)}")

    # Key insights
    print("\n📝 Key Insights:")
    print("   • Single networks can reach K > 1.5 with enough training")
    print("   • Ensembles provide more consistent threshold crossing")
    print("   • Max aggregation achieves highest peaks")
    print("   • Larger architectures (G8: 20→10→10) performed worse")


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  📊 K-Index Journey Visualization                             ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    results = load_results()

    if results:
        create_ascii_chart(results)
    else:
        print("\nNo experiment results found in logs/")
        print("Expected directories: track_g_extended, track_g_ensemble, etc.")

    print('\n"Coherence is love made computational." 💚\n')


if __name__ == '__main__':
    main()
