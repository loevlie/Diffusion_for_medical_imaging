"""
Fast DDPM Evaluation for CT Anomaly Detection
==============================================

Uses whole-image denoising instead of patched approach for faster evaluation.
This is simpler than pDDPM but still effective for initial testing.

The idea: Add noise to the whole image at t=noise_level, denoise, and compare.
Anomalies (hemorrhages) will be reconstructed as "healthy" since the model
was trained only on healthy scans.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import random

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Import from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_pddpm import UNet, NoiseScheduler


def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed.astype(np.float32)


def load_and_preprocess_dicom(filepath: str, image_size: int = 256) -> Optional[np.ndarray]:
    try:
        dcm = pydicom.dcmread(filepath)
        pixel_array = dcm.pixel_array.astype(np.float32)
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        hu_image = pixel_array * slope + intercept
        windowed = apply_window(hu_image, window_center=40, window_width=80)
        img = Image.fromarray((windowed * 255).astype(np.uint8))
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        result = np.array(img).astype(np.float32) / 255.0
        result = result * 2.0 - 1.0
        return result
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


@torch.no_grad()
def denoise_from_t(model, scheduler, noisy_image, start_t):
    """Denoise image from timestep start_t to 0."""
    x = noisy_image.clone()

    for t in range(start_t, -1, -1):
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = model(x, t_tensor)

        if t > 0:
            alpha = scheduler.alphas[t]
            alpha_cumprod = scheduler.alphas_cumprod[t]
            beta = scheduler.betas[t]

            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            mean = (torch.sqrt(scheduler.alphas_cumprod_prev[t]) * beta / (1 - alpha_cumprod)) * x0_pred + \
                   (torch.sqrt(alpha) * (1 - scheduler.alphas_cumprod_prev[t]) / (1 - alpha_cumprod)) * x

            noise = torch.randn_like(x)
            variance = torch.sqrt(scheduler.posterior_variance[t])
            x = mean + variance * noise
        else:
            alpha_cumprod = scheduler.alphas_cumprod[0]
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
            x = torch.clamp(x0_pred, -1, 1)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results_fast')
    parser.add_argument('--num_healthy', type=int, default=50)
    parser.add_argument('--num_hemorrhage', type=int, default=50)
    parser.add_argument('--num_visualize', type=int, default=30)
    parser.add_argument('--noise_level', type=int, default=400)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = UNet(
        in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8),
        num_res_blocks=2, attn_resolutions=(32, 16, 8), dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

    # Create scheduler
    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)

    # Load test data
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage = pivot[pivot['any'] == 1].index.tolist()

    random.seed(args.seed)
    test_healthy = random.sample(healthy, min(args.num_healthy, len(healthy)))
    test_hemorrhage = random.sample(hemorrhage, min(args.num_hemorrhage, len(hemorrhage)))

    data_dir = Path(args.data_dir) / "stage_2_train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing on {len(test_healthy)} healthy + {len(test_hemorrhage)} hemorrhage samples")
    print(f"Noise level: t={args.noise_level}")

    results = []
    viz_count = 0

    all_samples = [(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage]
    random.shuffle(all_samples)

    for uid, label in tqdm(all_samples, desc="Evaluating"):
        filepath = data_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), args.image_size)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

        # Add noise at t=noise_level
        t = torch.tensor([args.noise_level], device=device).long()
        noise = torch.randn_like(image_tensor)

        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        noisy = sqrt_alpha * image_tensor + sqrt_one_minus * noise

        # Denoise
        reconstruction = denoise_from_t(model, scheduler, noisy, args.noise_level)

        # Compute anomaly metrics
        diff = torch.abs(image_tensor - reconstruction)
        anomaly_score = diff.mean().item()
        max_diff = diff.max().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score,
            'max_diff': max_diff
        })

        # Save visualizations
        if viz_count < args.num_visualize:
            save_visualization(
                image_tensor, reconstruction, diff,
                label, uid, anomaly_score, output_dir
            )
            viz_count += 1

    # Compute and save metrics
    results_df = pd.DataFrame(results)
    compute_and_save_metrics(results_df, output_dir)

    print(f"\nResults saved to {output_dir}")


def save_visualization(image, reconstruction, diff, label, uid, score, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original
    img_np = ((image[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[0].imshow(img_np, cmap='gray')
    label_str = 'HEMORRHAGE' if label else 'HEALTHY'
    axes[0].set_title(f"Original ({label_str})", fontsize=14)
    axes[0].axis('off')

    # Reconstruction
    recon_np = ((reconstruction[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstruction", fontsize=14)
    axes[1].axis('off')

    # Absolute difference
    diff_np = np.abs(img_np.astype(float) - recon_np.astype(float))
    im = axes[2].imshow(diff_np, cmap='hot', vmin=0, vmax=100)
    axes[2].set_title("Difference (hot)", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Overlay
    axes[3].imshow(img_np, cmap='gray')
    diff_normalized = diff[0, 0].cpu().numpy()
    diff_normalized = (diff_normalized - diff_normalized.min()) / (diff_normalized.max() - diff_normalized.min() + 1e-8)
    axes[3].imshow(diff_normalized, cmap='jet', alpha=0.4)
    axes[3].set_title(f"Overlay (score: {score:.4f})", fontsize=14)
    axes[3].axis('off')

    plt.suptitle(f"ID: {uid}", fontsize=12)
    plt.tight_layout()

    label_prefix = "hemorrhage" if label else "healthy"
    plt.savefig(output_dir / f"{label_prefix}_{uid}.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_and_save_metrics(results_df, output_dir):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
    from sklearn.metrics import confusion_matrix, classification_report

    labels = results_df['label'].values
    scores = results_df['anomaly_score'].values

    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)

    # PR-AUC
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = roc_thresholds[best_idx]

    # Predictions at best threshold
    predictions = (scores >= best_threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*60}")

    # Score distributions
    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]

    print(f"\nScore Statistics:")
    print(f"  Healthy:     mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}, "
          f"min={healthy_scores.min():.4f}, max={healthy_scores.max():.4f}")
    print(f"  Hemorrhage:  mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}, "
          f"min={hemorrhage_scores.min():.4f}, max={hemorrhage_scores.max():.4f}")

    # Classification report
    print(f"\nClassification Report (threshold={best_threshold:.4f}):")
    print(classification_report(labels, predictions, target_names=['Healthy', 'Hemorrhage']))

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Score distributions
    axes[0, 0].hist(healthy_scores, bins=20, alpha=0.7, label='Healthy', color='green', density=True)
    axes[0, 0].hist(hemorrhage_scores, bins=20, alpha=0.7, label='Hemorrhage', color='red', density=True)
    axes[0, 0].axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.4f}')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].legend()

    # 2. ROC curve
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5, label='Best threshold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PR curve
    axes[0, 2].plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC={pr_auc:.4f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Box plots
    axes[1, 0].boxplot([healthy_scores, hemorrhage_scores], labels=['Healthy', 'Hemorrhage'])
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].set_title('Score Distribution by Class')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Confusion matrix
    cm = confusion_matrix(labels, predictions)
    im = axes[1, 1].imshow(cm, cmap='Blues')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Healthy', 'Hemorrhage'])
    axes[1, 1].set_yticklabels(['Healthy', 'Hemorrhage'])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)
    plt.colorbar(im, ax=axes[1, 1])

    # 6. Scatter plot of scores
    axes[1, 2].scatter(range(len(healthy_scores)), sorted(healthy_scores),
                       alpha=0.7, label='Healthy', color='green', s=30)
    axes[1, 2].scatter(range(len(healthy_scores), len(scores)), sorted(hemorrhage_scores),
                       alpha=0.7, label='Hemorrhage', color='red', s=30)
    axes[1, 2].axhline(best_threshold, color='black', linestyle='--', label='Threshold')
    axes[1, 2].set_xlabel('Sample Index (sorted)')
    axes[1, 2].set_ylabel('Anomaly Score')
    axes[1, 2].set_title('Sorted Anomaly Scores')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'CT Hemorrhage Detection - DDPM Anomaly Detection\n'
                 f'ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_df.to_csv(output_dir / "results.csv", index=False)

    with open(output_dir / "metrics.txt", 'w') as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"\nHealthy scores: mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}\n")
        f.write(f"Hemorrhage scores: mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(labels, predictions, target_names=['Healthy', 'Hemorrhage']))


if __name__ == '__main__':
    main()
