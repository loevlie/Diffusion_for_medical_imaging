#!/usr/bin/env python3
"""
One-Step pDDPM Evaluation - Following the paper's approach (arXiv:2303.03758)

KEY INSIGHT FROM PAPER:
- The paper uses ONE-STEP denoising, not iterative DDPM sampling
- At inference, they add noise at t=400, then predict x0 DIRECTLY in one step
- This is much faster and works because DDPM models can predict x0 from any noisy input

The formula is:
    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

To predict x0 from x_t:
    x0_pred = (x_t - sqrt(1 - alpha_cumprod_t) * predicted_noise) / sqrt(alpha_cumprod_t)

This is SINGLE forward pass - no iteration needed!
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import json

# Import model from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_pddpm import UNet


class NoiseScheduler:
    def __init__(self, timesteps=1000, schedule='cosine', device='cpu'):
        self.timesteps = timesteps
        self.device = device

        if schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999).to(device)
        else:
            self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)


def load_and_preprocess_dicom(filepath, image_size=256):
    """Load DICOM and preprocess with brain window."""
    try:
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)

        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept

        # Brain window
        window_center, window_width = 40, 80
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)

        # Resize
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
        result = np.array(pil_img).astype(np.float32) / 255.0

        # Normalize to [-1, 1]
        result = result * 2.0 - 1.0
        return result
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


@torch.no_grad()
def one_step_denoise(model, scheduler, noisy_image, t):
    """
    ONE-STEP denoising as described in the pDDPM paper.

    Given noisy image x_t at timestep t, predict x_0 directly.

    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

    Model predicts noise, so:
    x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * pred_noise) / sqrt(alpha_cumprod_t)
    """
    t_tensor = torch.full((noisy_image.shape[0],), t, device=noisy_image.device, dtype=torch.long)

    # Single forward pass - predict noise
    pred_noise = model(noisy_image, t_tensor)

    # Reconstruct x0 from noisy input and predicted noise
    sqrt_alpha_cumprod = scheduler.sqrt_alphas_cumprod[t]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t]

    x0_pred = (noisy_image - sqrt_one_minus * pred_noise) / sqrt_alpha_cumprod
    x0_pred = torch.clamp(x0_pred, -1, 1)

    return x0_pred


def main():
    parser = argparse.ArgumentParser(description='One-Step pDDPM Evaluation (Paper Method)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results_onestep')
    parser.add_argument('--num_healthy', type=int, default=100)
    parser.add_argument('--num_hemorrhage', type=int, default=100)
    parser.add_argument('--num_visualize', type=int, default=30)
    parser.add_argument('--noise_level', type=int, default=400,
                        help='Timestep for noise (paper uses 400)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("ONE-STEP DENOISING (Following pDDPM Paper)")
    print(f"{'='*60}")
    print(f"This is the method from arXiv:2303.03758")
    print(f"- Add noise at t={args.noise_level}")
    print(f"- Predict x0 in SINGLE forward pass (no iteration)")
    print(f"- Compare reconstruction to original")
    print(f"{'='*60}\n")

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

    for uid, label in tqdm(all_samples, desc="Evaluating (one-step)"):
        filepath = data_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), args.image_size)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

        # Add noise at t=noise_level
        t = args.noise_level
        noise = torch.randn_like(image_tensor)

        sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t]
        noisy = sqrt_alpha * image_tensor + sqrt_one_minus * noise

        # ONE-STEP denoising (this is the key difference from iterative DDPM)
        reconstruction = one_step_denoise(model, scheduler, noisy, t)

        # Compute anomaly metrics
        diff = torch.abs(image_tensor - reconstruction)
        anomaly_score = diff.mean().item()
        max_diff = diff.max().item()

        # Also compute MSE and SSIM-like metric
        mse = ((image_tensor - reconstruction) ** 2).mean().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score,
            'max_diff': max_diff,
            'mse': mse
        })

        # Save visualization
        if viz_count < args.num_visualize:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            orig = (image_tensor[0, 0].cpu().numpy() + 1) / 2
            noisy_img = (noisy[0, 0].cpu().numpy() + 1) / 2
            recon = (reconstruction[0, 0].cpu().numpy() + 1) / 2
            diff_img = diff[0, 0].cpu().numpy()

            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(noisy_img, cmap='gray')
            axes[1].set_title(f'Noisy (t={args.noise_level})')
            axes[1].axis('off')

            axes[2].imshow(recon, cmap='gray')
            axes[2].set_title('Reconstructed (ONE-STEP)')
            axes[2].axis('off')

            im = axes[3].imshow(diff_img, cmap='hot', vmin=0, vmax=0.5)
            axes[3].set_title(f'Difference (score={anomaly_score:.4f})')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046)

            label_str = "hemorrhage" if label == 1 else "healthy"
            plt.suptitle(f'{label_str.upper()} - One-Step Denoising', fontsize=14)
            plt.tight_layout()

            plt.savefig(output_dir / f'{label_str}_{uid[:9]}.png', dpi=100, bbox_inches='tight')
            plt.close()
            viz_count += 1

    # Calculate metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'results.csv', index=False)

    labels = np.array([r['label'] for r in results])
    scores = np.array([r['anomaly_score'] for r in results])

    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*60}")
    print("ONE-STEP DENOISING RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"")
    print(f"Healthy mean score:     {df_results[df_results['label']==0]['anomaly_score'].mean():.4f}")
    print(f"Hemorrhage mean score:  {df_results[df_results['label']==1]['anomaly_score'].mean():.4f}")
    print(f"{'='*60}")

    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ROC curve
    axes[0, 0].plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve (One-Step Denoising)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PR curve
    axes[0, 1].plot(recall, precision, 'g-', label=f'PR (AUC={pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Score distribution
    healthy_scores = df_results[df_results['label'] == 0]['anomaly_score']
    hemorrhage_scores = df_results[df_results['label'] == 1]['anomaly_score']

    axes[1, 0].hist(healthy_scores, bins=30, alpha=0.6, label='Healthy', color='green')
    axes[1, 0].hist(hemorrhage_scores, bins=30, alpha=0.6, label='Hemorrhage', color='red')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()

    # Box plot
    axes[1, 1].boxplot([healthy_scores, hemorrhage_scores], labels=['Healthy', 'Hemorrhage'])
    axes[1, 1].set_ylabel('Anomaly Score')
    axes[1, 1].set_title('Score by Class')

    plt.suptitle(f'One-Step pDDPM Evaluation (t={args.noise_level})\nROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'noise_level': args.noise_level,
        'method': 'one-step',
        'num_healthy': len(test_healthy),
        'num_hemorrhage': len(test_hemorrhage),
        'healthy_mean_score': float(healthy_scores.mean()),
        'hemorrhage_mean_score': float(hemorrhage_scores.mean())
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
