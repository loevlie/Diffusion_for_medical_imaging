"""
Evaluate pretrained/fine-tuned DDPM model for CT anomaly detection.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pydicom
import torch
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from train_pddpm_pretrained import load_model_for_inference, load_and_preprocess_dicom


@torch.no_grad()
def denoise_from_t(model, scheduler, noisy_image, start_t):
    """Denoise image from timestep start_t to 0."""
    x = noisy_image.clone()

    # Get relevant timesteps
    scheduler.set_timesteps(1000)
    relevant_timesteps = [t for t in scheduler.timesteps if t <= start_t]

    for t in relevant_timesteps:
        residual = model(x, t).sample
        x = scheduler.step(residual, t, x).prev_sample

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_pretrained/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results_pretrained')
    parser.add_argument('--num_healthy', type=int, default=50)
    parser.add_argument('--num_hemorrhage', type=int, default=50)
    parser.add_argument('--num_visualize', type=int, default=30)
    parser.add_argument('--noise_level', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model_for_inference(args.checkpoint, device)
    print(f"Config: {config}")

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.get('timesteps', 1000),
        beta_schedule="squaredcos_cap_v2"
    )

    image_size = config.get('image_size', 256)
    use_grayscale = config.get('adapt_channels', True)

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

        image = load_and_preprocess_dicom(str(filepath), image_size)
        if image is None:
            continue

        # Prepare tensor
        if use_grayscale:
            image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        else:
            # Replicate to 3 channels
            image_tensor = torch.tensor(np.stack([image]*3, axis=0)).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Add noise at t=noise_level
        t = torch.tensor([args.noise_level], device=device).long()
        noise = torch.randn_like(image_tensor)
        noisy = scheduler.add_noise(image_tensor, noise, t)

        # Denoise
        reconstruction = denoise_from_t(model, scheduler, noisy, args.noise_level)

        # Compute anomaly metrics
        diff = torch.abs(image_tensor - reconstruction)
        anomaly_score = diff.mean().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score,
        })

        # Save visualizations
        if viz_count < args.num_visualize:
            save_visualization(
                image_tensor, reconstruction, diff,
                label, uid, anomaly_score, output_dir, use_grayscale
            )
            viz_count += 1

    # Compute metrics
    results_df = pd.DataFrame(results)
    compute_metrics(results_df, output_dir)

    print(f"\nResults saved to {output_dir}")


def save_visualization(image, reconstruction, diff, label, uid, score, output_dir, use_grayscale):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    if use_grayscale:
        img_np = ((image[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        recon_np = ((reconstruction[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        diff_np = diff[0, 0].cpu().numpy()
    else:
        img_np = ((image[0].mean(dim=0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        recon_np = ((reconstruction[0].mean(dim=0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        diff_np = diff[0].mean(dim=0).cpu().numpy()

    axes[0].imshow(img_np, cmap='gray')
    label_str = 'HEMORRHAGE' if label else 'HEALTHY'
    axes[0].set_title(f"Original ({label_str})", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstruction", fontsize=14)
    axes[1].axis('off')

    im = axes[2].imshow(np.abs(img_np.astype(float) - recon_np.astype(float)), cmap='hot', vmin=0, vmax=100)
    axes[2].set_title("Difference", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(img_np, cmap='gray')
    diff_normalized = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)
    axes[3].imshow(diff_normalized, cmap='jet', alpha=0.4)
    axes[3].set_title(f"Overlay (score: {score:.4f})", fontsize=14)
    axes[3].axis('off')

    plt.suptitle(f"ID: {uid}", fontsize=12)
    plt.tight_layout()

    label_prefix = "hemorrhage" if label else "healthy"
    plt.savefig(output_dir / f"{label_prefix}_{uid}.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(results_df, output_dir):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, classification_report, confusion_matrix

    labels = results_df['label'].values
    scores = results_df['anomaly_score'].values

    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = roc_thresholds[best_idx]
    predictions = (scores >= best_threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*60}")

    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]
    print(f"\nHealthy:     mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}")
    print(f"Hemorrhage:  mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}")

    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Healthy', 'Hemorrhage']))

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].hist(healthy_scores, bins=20, alpha=0.7, label='Healthy', color='green', density=True)
    axes[0, 0].hist(hemorrhage_scores, bins=20, alpha=0.7, label='Hemorrhage', color='red', density=True)
    axes[0, 0].axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.4f}')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].legend()

    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5)
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC={pr_auc:.4f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].boxplot([healthy_scores, hemorrhage_scores], labels=['Healthy', 'Hemorrhage'])
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].set_title('Score Distribution by Class')

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

    axes[1, 2].scatter(range(len(healthy_scores)), sorted(healthy_scores), alpha=0.7, label='Healthy', color='green', s=30)
    axes[1, 2].scatter(range(len(healthy_scores), len(scores)), sorted(hemorrhage_scores), alpha=0.7, label='Hemorrhage', color='red', s=30)
    axes[1, 2].axhline(best_threshold, color='black', linestyle='--', label='Threshold')
    axes[1, 2].set_xlabel('Sample Index (sorted)')
    axes[1, 2].set_ylabel('Anomaly Score')
    axes[1, 2].set_title('Sorted Anomaly Scores')
    axes[1, 2].legend()

    plt.suptitle(f'CT Hemorrhage Detection - Pretrained DDPM\nROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    results_df.to_csv(output_dir / "results.csv", index=False)

    with open(output_dir / "metrics.txt", 'w') as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"\nHealthy: mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}\n")
        f.write(f"Hemorrhage: mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}\n")


if __name__ == '__main__':
    main()
