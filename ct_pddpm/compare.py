"""
Compare Scratch vs Pretrained Training Curves
==============================================

Generates a comparison plot showing:
1. ROC-AUC over epochs for both models
2. Validation loss over epochs for both models
3. Anomaly score separation
4. Summary statistics table

Usage:
    python compare.py
    python compare.py --scratch_dir checkpoints_patched --pretrained_dir checkpoints_pretrained_3ch
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_history(checkpoint_dir):
    """Load training history from checkpoint directory."""
    history_path = Path(checkpoint_dir) / 'training_history.json'
    if not history_path.exists():
        return None
    with open(history_path) as f:
        return json.load(f)


def plot_comparison(scratch_hist, pretrained_hist, output_path='comparison.png',
                    scratch_name='From Scratch', pretrained_name='Pretrained HF'):
    """Generate comparison plot like the reference image."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Determine epochs
    scratch_epochs = list(range(1, len(scratch_hist['roc_auc']) + 1)) if scratch_hist else []
    pretrained_epochs = list(range(1, len(pretrained_hist['roc_auc']) + 1)) if pretrained_hist else []

    # ===== Plot 1: ROC-AUC =====
    ax1 = axes[0, 0]

    if scratch_hist:
        scratch_best = max(scratch_hist['roc_auc'])
        scratch_best_ep = scratch_hist['roc_auc'].index(scratch_best) + 1
        ax1.plot(scratch_epochs, scratch_hist['roc_auc'], 'g-o', linewidth=2, markersize=5,
                 label=f"{scratch_name} (Best: {scratch_best:.3f} @ ep{scratch_best_ep})")
        ax1.axhline(y=scratch_best, color='g', linestyle='--', alpha=0.5)

    if pretrained_hist:
        pretrained_best = max(pretrained_hist['roc_auc'])
        pretrained_best_ep = pretrained_hist['roc_auc'].index(pretrained_best) + 1
        ax1.plot(pretrained_epochs, pretrained_hist['roc_auc'], 'b-s', linewidth=2, markersize=5,
                 label=f"{pretrained_name} (Best: {pretrained_best:.3f} @ ep{pretrained_best_ep})")
        ax1.axhline(y=pretrained_best, color='b', linestyle='--', alpha=0.5)

    # Add shaded region showing improvement
    if scratch_hist and pretrained_hist:
        scratch_best = max(scratch_hist['roc_auc'])
        pretrained_best = max(pretrained_hist['roc_auc'])
        if scratch_best > pretrained_best:
            improvement = (scratch_best - pretrained_best) / pretrained_best * 100
            max_epoch = max(len(scratch_epochs), len(pretrained_epochs))
            ax1.fill_between(range(1, max_epoch + 1), pretrained_best, scratch_best,
                           alpha=0.15, color='g')
            ax1.text(max_epoch * 0.75, (scratch_best + pretrained_best) / 2,
                    f'+{improvement:.1f}%\nimprovement',
                    fontsize=11, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('ROC-AUC', fontsize=12)
    ax1.set_title('Anomaly Detection Performance (ROC-AUC)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.5, 0.8)

    # ===== Plot 2: Validation Loss =====
    ax2 = axes[0, 1]

    if scratch_hist:
        ax2.plot(scratch_epochs, scratch_hist['val_loss'], 'g-o', linewidth=2, markersize=5,
                 label=scratch_name)

    if pretrained_hist:
        ax2.plot(pretrained_epochs, pretrained_hist['val_loss'], 'b-s', linewidth=2, markersize=5,
                 label=pretrained_name)

    # Add annotation
    ax2.annotate('Lower loss ≠ Better detection!',
                xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax2.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)

    # ===== Plot 3: Score Separation =====
    ax3 = axes[1, 0]

    models = []
    healthy_scores = []
    hemorrhage_scores = []
    colors = []

    if scratch_hist:
        models.append(scratch_name.replace(' ', '\n'))
        healthy_scores.append(scratch_hist['healthy_mean'][-1])
        hemorrhage_scores.append(scratch_hist['hemorrhage_mean'][-1])
        colors.append('green')

    if pretrained_hist:
        models.append(pretrained_name.replace(' ', '\n'))
        healthy_scores.append(pretrained_hist['healthy_mean'][-1])
        hemorrhage_scores.append(pretrained_hist['hemorrhage_mean'][-1])
        colors.append('blue')

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax3.bar(x - width/2, healthy_scores, width, label='Healthy',
                    color='lightgreen', edgecolor='green', linewidth=2)
    bars2 = ax3.bar(x + width/2, hemorrhage_scores, width, label='Hemorrhage',
                    color='lightcoral', edgecolor='red', linewidth=2)

    ax3.set_ylabel('Mean Anomaly Score', fontsize=12)
    ax3.set_title('Final Anomaly Score Separation', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')

    # Add separation annotations
    for i, (h, hem) in enumerate(zip(healthy_scores, hemorrhage_scores)):
        diff = hem - h
        max_val = max(h, hem)
        ax3.annotate(f'Δ = {diff:.4f}', xy=(i, max_val + 0.002),
                    ha='center', fontsize=10, fontweight='bold')

    # ===== Plot 4: Summary Table =====
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Prepare table data
    headers = ['Metric', scratch_name, pretrained_name]
    table_data = []

    if scratch_hist and pretrained_hist:
        scratch_best = max(scratch_hist['roc_auc'])
        scratch_best_ep = scratch_hist['roc_auc'].index(scratch_best) + 1
        pretrained_best = max(pretrained_hist['roc_auc'])
        pretrained_best_ep = pretrained_hist['roc_auc'].index(pretrained_best) + 1

        table_data = [
            ['Best ROC-AUC', f'{scratch_best:.4f}', f'{pretrained_best:.4f}'],
            ['Best Epoch', str(scratch_best_ep), str(pretrained_best_ep)],
            ['Total Epochs', str(len(scratch_hist['roc_auc'])), str(len(pretrained_hist['roc_auc']))],
            ['Final Val Loss', f"{scratch_hist['val_loss'][-1]:.6f}", f"{pretrained_hist['val_loss'][-1]:.6f}"],
            ['Healthy Mean', f"{scratch_hist['healthy_mean'][-1]:.6f}", f"{pretrained_hist['healthy_mean'][-1]:.6f}"],
            ['Hemorrhage Mean', f"{scratch_hist['hemorrhage_mean'][-1]:.6f}", f"{pretrained_hist['hemorrhage_mean'][-1]:.6f}"],
            ['Score Separation', f"{scratch_hist['hemorrhage_mean'][-1] - scratch_hist['healthy_mean'][-1]:.6f}",
             f"{pretrained_hist['hemorrhage_mean'][-1] - pretrained_hist['healthy_mean'][-1]:.6f}"],
            ['Input Channels', '1 (brain)', '3 (brain/subdural/bone)'],
            ['Parameters', '~14M', '113.7M'],
        ]

    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight winner cells
    if scratch_hist and pretrained_hist:
        # ROC-AUC row (row 1)
        if scratch_best > pretrained_best:
            table[(1, 1)].set_facecolor('#90EE90')  # Light green
        else:
            table[(1, 2)].set_facecolor('#90EE90')

    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('pDDPM for CT Hemorrhage Detection:\nFrom-Scratch vs Pretrained Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")


def print_summary(scratch_hist, pretrained_hist, scratch_name, pretrained_name):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    if scratch_hist:
        best_auc = max(scratch_hist['roc_auc'])
        best_ep = scratch_hist['roc_auc'].index(best_auc) + 1
        sep = scratch_hist['hemorrhage_mean'][-1] - scratch_hist['healthy_mean'][-1]
        print(f"\n{scratch_name}:")
        print(f"  Best ROC-AUC: {best_auc:.4f} (epoch {best_ep})")
        print(f"  Final val loss: {scratch_hist['val_loss'][-1]:.6f}")
        print(f"  Healthy mean:    {scratch_hist['healthy_mean'][-1]:.6f}")
        print(f"  Hemorrhage mean: {scratch_hist['hemorrhage_mean'][-1]:.6f}")
        print(f"  Separation:      {sep:.6f}")

    if pretrained_hist:
        best_auc = max(pretrained_hist['roc_auc'])
        best_ep = pretrained_hist['roc_auc'].index(best_auc) + 1
        sep = pretrained_hist['hemorrhage_mean'][-1] - pretrained_hist['healthy_mean'][-1]
        print(f"\n{pretrained_name}:")
        print(f"  Best ROC-AUC: {best_auc:.4f} (epoch {best_ep})")
        print(f"  Final val loss: {pretrained_hist['val_loss'][-1]:.6f}")
        print(f"  Healthy mean:    {pretrained_hist['healthy_mean'][-1]:.6f}")
        print(f"  Hemorrhage mean: {pretrained_hist['hemorrhage_mean'][-1]:.6f}")
        print(f"  Separation:      {sep:.6f}")

    if scratch_hist and pretrained_hist:
        scratch_best = max(scratch_hist['roc_auc'])
        pretrained_best = max(pretrained_hist['roc_auc'])
        improvement = (scratch_best - pretrained_best) / pretrained_best * 100
        print(f"\n{'='*70}")
        if scratch_best > pretrained_best:
            print(f"WINNER: {scratch_name} (+{improvement:.1f}% ROC-AUC improvement)")
        else:
            print(f"WINNER: {pretrained_name} (+{-improvement:.1f}% ROC-AUC improvement)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare scratch vs pretrained models')
    parser.add_argument('--scratch_dir', type=str, default='checkpoints_patched')
    parser.add_argument('--pretrained_dir', type=str, default='checkpoints_pretrained_3ch')
    parser.add_argument('--output', type=str, default='comparison.png')
    parser.add_argument('--scratch_name', type=str, default='From Scratch (1ch)')
    parser.add_argument('--pretrained_name', type=str, default='Pretrained HF (3ch)')
    parser.add_argument('--max_epochs', type=int, default=10, help='Limit to first N epochs')
    args = parser.parse_args()

    scratch_hist = load_history(args.scratch_dir)
    pretrained_hist = load_history(args.pretrained_dir)

    # Limit to max_epochs
    if scratch_hist and args.max_epochs:
        for key in scratch_hist:
            scratch_hist[key] = scratch_hist[key][:args.max_epochs]
    if pretrained_hist and args.max_epochs:
        for key in pretrained_hist:
            pretrained_hist[key] = pretrained_hist[key][:args.max_epochs]

    if not scratch_hist and not pretrained_hist:
        print("Error: No training history found in either directory")
        print(f"  Scratch dir: {args.scratch_dir}")
        print(f"  Pretrained dir: {args.pretrained_dir}")
        print("\nRun training first with: python train.py --output_dir <dir>")
        return

    if scratch_hist:
        print(f"Loaded {args.scratch_name}: {len(scratch_hist['roc_auc'])} epochs")
    else:
        print(f"{args.scratch_name}: No history found")

    if pretrained_hist:
        print(f"Loaded {args.pretrained_name}: {len(pretrained_hist['roc_auc'])} epochs")
    else:
        print(f"{args.pretrained_name}: No history found")

    plot_comparison(scratch_hist, pretrained_hist, args.output,
                   args.scratch_name, args.pretrained_name)
    print_summary(scratch_hist, pretrained_hist, args.scratch_name, args.pretrained_name)


if __name__ == "__main__":
    main()
