#!/usr/bin/env python3
"""
Patched One-Step pDDPM Evaluation - Following the paper's approach (arXiv:2303.03758)

KEY INSIGHTS FROM PAPER:
1. Use PATCHED approach: Add noise only to a patch, keep surrounding context clean
2. Use ONE-STEP denoising with objective='pred_x0' (model directly predicts clean image)
3. Optimal noise level: t=350-400 (high enough to hide anomalies)
4. Sliding window technique with context from adjacent areas

The patch-based approach:
- For each patch location, create an image where:
  - The target patch has noise added at t=350
  - Surrounding context remains CLEAN (original image)
- The model uses clean context to reconstruct the noisy patch
- Model directly predicts x0 (clean image), NOT noise
- Anomalies will have higher reconstruction error because the model
  only knows healthy anatomy
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
from opensimplex import OpenSimplex

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_pddpm import UNet


# =============================================================================
# Simplex Noise for pDDPM (following AnoDDPM paper)
# =============================================================================

class SimplexNoise:
    """Generate simplex noise for diffusion models (AnoDDPM style)."""

    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2**31)
        self.simplex = OpenSimplex(seed=seed)

    def generate_octave_noise_2d(self, shape, octaves=4, persistence=0.5, base_scale=32.0):
        """Generate multi-octave simplex noise."""
        height, width = shape
        noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        max_amp = 0.0

        for octave in range(octaves):
            scale = base_scale / (2 ** octave)
            for y in range(height):
                for x in range(width):
                    noise[y, x] += amplitude * self.simplex.noise2(x / scale, y / scale)
            max_amp += amplitude
            amplitude *= persistence

        noise /= max_amp
        return noise


def generate_simplex_noise_patch(channels, height, width, octaves=4, device='cpu'):
    """Generate simplex noise for a single patch, normalized to std=1.0."""
    simplex = SimplexNoise(seed=random.randint(0, 2**31))
    noise_channels = []

    for _ in range(channels):
        noise = simplex.generate_octave_noise_2d(
            (height, width),
            octaves=octaves,
            persistence=0.5,
            base_scale=32.0
        )
        noise_channels.append(noise)

    noise_tensor = torch.tensor(np.stack(noise_channels, axis=0), dtype=torch.float32)

    # CRITICAL: Normalize to std=1.0 like Gaussian noise
    noise_std = noise_tensor.std()
    if noise_std > 0:
        noise_tensor = noise_tensor / noise_std

    return noise_tensor.unsqueeze(0).to(device)  # Add batch dimension


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


def load_and_preprocess_dicom(filepath, image_size=256, center_on_brain=True):
    """Load DICOM and preprocess with brain window, optionally centering on brain."""
    try:
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)

        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept

        # Store raw HU for brain detection
        img_hu = img.copy()

        # Brain window
        window_center, window_width = 40, 80
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)

        # Center on brain region if requested (conservative cropping)
        if center_on_brain:
            # Include skull in mask to avoid cropping too tight
            brain_mask = (img_hu > -50) & (img_hu < 150)

            from scipy import ndimage
            if brain_mask.any():
                # Clean up mask - be generous
                brain_mask = ndimage.binary_closing(brain_mask, iterations=3)
                brain_mask = ndimage.binary_dilation(brain_mask, iterations=10)

                # Find largest connected component
                labeled, num_features = ndimage.label(brain_mask)
                if num_features > 0:
                    component_sizes = ndimage.sum(brain_mask, labeled, range(1, num_features + 1))
                    largest_idx = np.argmax(component_sizes) + 1
                    brain_mask = labeled == largest_idx

                    # Get bounding box
                    rows = np.any(brain_mask, axis=1)
                    cols = np.any(brain_mask, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]

                        # Add generous padding
                        pad = 25
                        rmin = max(0, rmin - pad)
                        rmax = min(img.shape[0], rmax + pad)
                        cmin = max(0, cmin - pad)
                        cmax = min(img.shape[1], cmax + pad)

                        crop_h = rmax - rmin
                        crop_w = cmax - cmin
                        orig_size = min(img.shape)

                        # Only crop if keeping at least 70% of original
                        if crop_h >= orig_size * 0.7 and crop_w >= orig_size * 0.7:
                            img = img[rmin:rmax, cmin:cmax]

        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
        result = np.array(pil_img).astype(np.float32) / 255.0

        result = result * 2.0 - 1.0
        return result
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


@torch.no_grad()
def one_step_denoise(model, scheduler, noisy_image, t):
    """
    ONE-STEP denoising: model predicts x_0 directly from x_t.

    Following the paper's approach (objective='pred_x0'), the model
    directly outputs the reconstructed clean image, not the noise.
    """
    t_tensor = torch.full((noisy_image.shape[0],), t, device=noisy_image.device, dtype=torch.long)

    # Model directly predicts x0 (paper's approach)
    x0_pred = model(noisy_image, t_tensor)
    x0_pred = torch.clamp(x0_pred, -1, 1)

    return x0_pred


@torch.no_grad()
def patched_anomaly_detection(model, scheduler, image_tensor, patch_size, stride, noise_level, device, noise_type='simplex'):
    """
    Patched pDDPM anomaly detection as described in the paper.

    For each patch:
    1. Keep the full image as context
    2. Add noise ONLY to the target patch
    3. Run one-step denoising on the full image
    4. Extract the reconstructed patch
    5. Compare to original patch

    This way, the model uses clean surrounding context to help reconstruct
    the noisy patch, and anomalies (which the model hasn't seen) will have
    higher reconstruction error.
    """
    B, C, H, W = image_tensor.shape

    # Create anomaly score map
    anomaly_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    # Get noise schedule parameters
    sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

    # Sliding window over patches
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Create a copy of the original image
            patched_input = image_tensor.clone()

            # Generate noise for just the patch
            if noise_type == 'simplex':
                patch_noise = generate_simplex_noise_patch(C, patch_size, patch_size, device=device)
            else:
                patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)

            # Add noise ONLY to the target patch (context stays clean!)
            original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            # One-step denoising of the full image
            reconstructed = one_step_denoise(model, scheduler, patched_input, noise_level)

            # Extract reconstructed patch and compute error
            recon_patch = reconstructed[:, :, y:y+patch_size, x:x+patch_size]
            patch_error = torch.abs(original_patch - recon_patch).squeeze()

            # Accumulate error in anomaly map
            anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
            count_map[y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    count_map = torch.clamp(count_map, min=1)
    anomaly_map = anomaly_map / count_map

    return anomaly_map


def main():
    parser = argparse.ArgumentParser(description='Patched One-Step pDDPM Evaluation')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results_patched')
    parser.add_argument('--num_healthy', type=int, default=50)
    parser.add_argument('--num_hemorrhage', type=int, default=50)
    parser.add_argument('--num_visualize', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size (paper uses ~60)')
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride for sliding window (smaller = more overlap)')
    parser.add_argument('--noise_level', type=int, default=350,
                        help='Timestep for noise (paper recommends 350)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--noise_type', type=str, default='simplex', choices=['simplex', 'gaussian'],
                        help='Type of noise to use (default: simplex, per paper)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("PATCHED ONE-STEP pDDPM (Following Paper Method)")
    print(f"{'='*60}")
    print(f"Paper: arXiv:2303.03758")
    print(f"Method:")
    print(f"  1. Slide {args.patch_size}x{args.patch_size} window over image")
    print(f"  2. Add noise (t={args.noise_level}) ONLY to patch, keep context clean")
    print(f"  3. One-step denoise using context to guide reconstruction")
    print(f"  4. Anomalies = high reconstruction error")
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

    # Calculate number of patches
    num_patches_per_dim = (args.image_size - args.patch_size) // args.stride + 1
    total_patches = num_patches_per_dim ** 2
    print(f"Testing on {len(test_healthy)} healthy + {len(test_hemorrhage)} hemorrhage samples")
    print(f"Patch size: {args.patch_size}, Stride: {args.stride}")
    print(f"Patches per image: {total_patches} ({num_patches_per_dim}x{num_patches_per_dim})")
    print(f"Noise level: t={args.noise_level}")
    print(f"Noise type: {args.noise_type}")

    results = []
    viz_count = 0

    all_samples = [(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage]
    random.shuffle(all_samples)

    for uid, label in tqdm(all_samples, desc="Evaluating (patched)"):
        filepath = data_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), args.image_size)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

        # Run patched anomaly detection
        anomaly_map = patched_anomaly_detection(
            model, scheduler, image_tensor,
            args.patch_size, args.stride, args.noise_level, device,
            noise_type=args.noise_type
        )

        # Compute scores
        anomaly_score = anomaly_map.mean().item()
        max_score = anomaly_map.max().item()

        # Also compute score focusing on brain region (center of image)
        center_region = anomaly_map[64:192, 64:192]  # Center 128x128
        center_score = center_region.mean().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score,
            'max_score': max_score,
            'center_score': center_score
        })

        # Save visualization showing the full patch grid approach
        if viz_count < args.num_visualize:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            orig = (image_tensor[0, 0].cpu().numpy() + 1) / 2
            amap = anomaly_map.cpu().numpy()

            # Panel 1: Original with patch grid overlay
            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title('Original + Patch Grid')
            axes[0].axis('off')
            # Draw patch grid to show sliding window scheme
            for y in range(0, args.image_size - args.patch_size + 1, args.stride):
                for x in range(0, args.image_size - args.patch_size + 1, args.stride):
                    rect = plt.Rectangle((x, y), args.patch_size, args.patch_size,
                                        fill=False, color='cyan', linewidth=0.5, alpha=0.5)
                    axes[0].add_patch(rect)

            # Panel 2: Example of one noisy patch (center)
            ps = args.patch_size
            cy, cx = args.image_size // 2, args.image_size // 2
            y1, x1 = cy - ps//2, cx - ps//2

            sqrt_alpha = scheduler.sqrt_alphas_cumprod[args.noise_level]
            sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[args.noise_level]

            patched_input = image_tensor.clone()
            if args.noise_type == 'simplex':
                patch_noise = generate_simplex_noise_patch(1, ps, ps, device=device)
            else:
                patch_noise = torch.randn(1, 1, ps, ps, device=device)
            original_patch = image_tensor[:, :, y1:y1+ps, x1:x1+ps].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y1:y1+ps, x1:x1+ps] = noisy_patch

            noisy_img = (patched_input[0, 0].cpu().numpy() + 1) / 2
            axes[1].imshow(noisy_img, cmap='gray')
            axes[1].set_title(f'Example: Noisy Patch (t={args.noise_level})')
            axes[1].axis('off')
            rect = plt.Rectangle((x1, y1), ps, ps, fill=False, color='red', linewidth=2)
            axes[1].add_patch(rect)

            # Panel 3: Full anomaly map from ALL patches
            im = axes[2].imshow(amap, cmap='hot', vmin=0, vmax=0.3)
            axes[2].set_title(f'Anomaly Map (all {num_patches_per_dim}x{num_patches_per_dim} patches)')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            # Panel 4: Overlay
            axes[3].imshow(orig, cmap='gray')
            axes[3].imshow(amap, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
            axes[3].set_title(f'Overlay (score={anomaly_score:.4f})')
            axes[3].axis('off')

            label_str = "hemorrhage" if label == 1 else "healthy"
            plt.suptitle(f'{label_str.upper()} - Patched pDDPM (t={args.noise_level}, {args.patch_size}x{args.patch_size} patches, stride={args.stride})',
                         fontsize=14)
            plt.tight_layout()

            plt.savefig(output_dir / f'{label_str}_{uid[:9]}.png', dpi=100, bbox_inches='tight')
            plt.close()
            viz_count += 1

    # Calculate metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'results.csv', index=False)

    labels = np.array([r['label'] for r in results])

    # Try different score types
    for score_name in ['anomaly_score', 'max_score', 'center_score']:
        scores = np.array([r[score_name] for r in results])

        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        print(f"\n{score_name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    # Use main anomaly_score for final metrics
    scores = np.array([r['anomaly_score'] for r in results])
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*60}")
    print("PATCHED ONE-STEP pDDPM RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"")
    print(f"Healthy mean score:     {df_results[df_results['label']==0]['anomaly_score'].mean():.4f}")
    print(f"Hemorrhage mean score:  {df_results[df_results['label']==1]['anomaly_score'].mean():.4f}")
    print(f"{'='*60}")

    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve (Patched One-Step)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(recall, precision, 'g-', label=f'PR (AUC={pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    healthy_scores = df_results[df_results['label'] == 0]['anomaly_score']
    hemorrhage_scores = df_results[df_results['label'] == 1]['anomaly_score']

    axes[1, 0].hist(healthy_scores, bins=20, alpha=0.6, label='Healthy', color='green')
    axes[1, 0].hist(hemorrhage_scores, bins=20, alpha=0.6, label='Hemorrhage', color='red')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()

    axes[1, 1].boxplot([healthy_scores, hemorrhage_scores], labels=['Healthy', 'Hemorrhage'])
    axes[1, 1].set_ylabel('Anomaly Score')
    axes[1, 1].set_title('Score by Class')

    plt.suptitle(f'Patched pDDPM Evaluation (t={args.noise_level}, patch={args.patch_size})\n'
                 f'ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'noise_level': args.noise_level,
        'patch_size': args.patch_size,
        'stride': args.stride,
        'method': 'patched_one_step',
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
