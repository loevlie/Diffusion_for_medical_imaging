#!/usr/bin/env python3
"""
Patched One-Step pDDPM Evaluation for CT Hemorrhage Detection

Based on arXiv:2303.03758 - Patched Diffusion Models for Unsupervised Anomaly Detection

Method:
1. Add noise ONLY to a patch, keep surrounding context clean
2. One-step denoising: model predicts noise, compute x0
3. Sliding window over image to build anomaly map
4. Anomalies = high reconstruction error (model hasn't seen hemorrhages)

Supports both:
- Custom UNet (from train_from_scratch.py)
- HuggingFace Diffusers UNet2DModel (from train_from_pretrained.py)
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_from_scratch import UNet

# Flag for HuggingFace model
IS_HF_MODEL = False


class NoiseScheduler:
    """Cosine noise schedule for DDPM."""

    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        # Cosine schedule
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999).to(device)

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

        # Brain window (W:80, L:40)
        window_center, window_width = 40, 80
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)

        # Resize
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        img = np.array(pil_img).astype(np.float32) / 255.0

        # Normalize to [-1, 1]
        img = img * 2 - 1
        return img

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


@torch.no_grad()
def one_step_denoise(model, scheduler, noisy_image, t, is_hf_model=False):
    """
    One-step denoising: predict noise, then compute x0.
    x0 = (x_t - sqrt(1-alpha_cumprod) * pred_noise) / sqrt(alpha_cumprod)

    Args:
        is_hf_model: If True, model is a HuggingFace UNet2DModel that returns a dict
    """
    t_tensor = torch.full((noisy_image.shape[0],), t, device=noisy_image.device, dtype=torch.long)

    output = model(noisy_image, t_tensor)

    # HuggingFace models return a dict with 'sample' key
    if is_hf_model and hasattr(output, 'sample'):
        pred_noise = output.sample
    else:
        pred_noise = output

    sqrt_alpha = scheduler.sqrt_alphas_cumprod[t]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t]

    x0_pred = (noisy_image - sqrt_one_minus * pred_noise) / sqrt_alpha
    x0_pred = torch.clamp(x0_pred, -1, 1)

    return x0_pred


@torch.no_grad()
def patched_anomaly_detection(model, scheduler, image_tensor, patch_size, stride, noise_level, device, is_hf_model=False):
    """
    Patched pDDPM anomaly detection.

    For each patch:
    1. Add noise ONLY to the patch (context stays clean)
    2. One-step denoise
    3. Compute reconstruction error on patch
    """
    B, C, H, W = image_tensor.shape

    anomaly_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

    # Sliding window
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Create noisy input (only patch is noisy)
            patched_input = image_tensor.clone()
            patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)

            original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            # One-step denoise
            reconstructed = one_step_denoise(model, scheduler, patched_input, noise_level, is_hf_model)

            # Compute error on patch
            recon_patch = reconstructed[:, :, y:y+patch_size, x:x+patch_size]
            patch_error = torch.abs(original_patch - recon_patch).squeeze()

            anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
            count_map[y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    count_map = torch.clamp(count_map, min=1)
    anomaly_map = anomaly_map / count_map

    return anomaly_map


def main():
    parser = argparse.ArgumentParser(description='Patched pDDPM Evaluation for CT Hemorrhage Detection')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--num_healthy', type=int, default=50)
    parser.add_argument('--num_hemorrhage', type=int, default=50)
    parser.add_argument('--num_visualize', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--noise_level', type=int, default=350)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nPatched pDDPM Evaluation")
    print(f"=" * 50)
    print(f"Patch size: {args.patch_size}, Stride: {args.stride}")
    print(f"Noise level: t={args.noise_level}")
    print(f"=" * 50)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Check if this is a HuggingFace model checkpoint
    is_hf_model = 'hf_model' in checkpoint or 'config' in checkpoint

    if is_hf_model:
        # Load HuggingFace model
        from diffusers import UNet2DModel
        from train_from_pretrained import adapt_unet_for_grayscale

        hf_model_id = checkpoint.get('hf_model', 'google/ddpm-ema-celebahq-256')
        print(f"Detected HuggingFace model checkpoint (base: {hf_model_id})")

        # Create model with same config
        config = checkpoint.get('config', {})
        if config.get('in_channels', 3) == 1:
            # Already adapted, create directly from config
            model = UNet2DModel(**config).to(device)
        else:
            # Need to adapt from base model
            model = UNet2DModel.from_pretrained(hf_model_id)
            model = adapt_unet_for_grayscale(model)
            model = model.to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        loss_info = checkpoint.get('val_loss', checkpoint.get('train_loss', checkpoint.get('loss', 'N/A')))
        print(f"Loaded HuggingFace model from epoch {checkpoint['epoch']} with loss {loss_info}")
    else:
        # Load custom UNet
        model = UNet(
            in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8),
            num_res_blocks=2, attn_resolutions=(32, 16, 8), dropout=0.0
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        loss_info = checkpoint.get('val_loss', checkpoint.get('loss', 'N/A'))
        print(f"Loaded custom UNet from epoch {checkpoint['epoch']} with loss {loss_info}")

    # Create scheduler
    scheduler = NoiseScheduler(timesteps=1000, device=device)

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

        # Run patched anomaly detection
        anomaly_map = patched_anomaly_detection(
            model, scheduler, image_tensor,
            args.patch_size, args.stride, args.noise_level, device, is_hf_model
        )

        # Compute scores
        anomaly_score = anomaly_map.mean().item()

        # Center score (center 64x64 region)
        center = args.image_size // 2
        half = 32
        center_score = anomaly_map[center-half:center+half, center-half:center+half].mean().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score,
            'center_score': center_score,
        })

        # Visualize
        if viz_count < args.num_visualize:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            orig = (image + 1) / 2
            amap = anomaly_map.cpu().numpy()

            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Show noisy patch example
            ps = args.patch_size
            cy, cx = args.image_size // 2, args.image_size // 2
            y1, x1 = cy - ps//2, cx - ps//2

            sqrt_alpha = scheduler.sqrt_alphas_cumprod[args.noise_level]
            sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[args.noise_level]

            patched_input = image_tensor.clone()
            patch_noise = torch.randn(1, 1, ps, ps, device=device)
            original_patch = image_tensor[:, :, y1:y1+ps, x1:x1+ps].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y1:y1+ps, x1:x1+ps] = noisy_patch

            noisy_img = (patched_input[0, 0].cpu().numpy() + 1) / 2
            axes[1].imshow(noisy_img, cmap='gray')
            axes[1].set_title(f'Noisy Patch (t={args.noise_level})')
            axes[1].axis('off')
            rect = plt.Rectangle((x1, y1), ps, ps, fill=False, color='red', linewidth=2)
            axes[1].add_patch(rect)

            im = axes[2].imshow(amap, cmap='hot', vmin=0, vmax=0.3)
            axes[2].set_title('Anomaly Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046)

            axes[3].imshow(orig, cmap='gray')
            axes[3].imshow(amap, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
            axes[3].set_title(f'Overlay (score={anomaly_score:.4f})')
            axes[3].axis('off')

            label_str = "hemorrhage" if label == 1 else "healthy"
            plt.suptitle(f'{label_str.upper()} - Patched pDDPM', fontsize=14)
            plt.tight_layout()

            plt.savefig(output_dir / f'{label_str}_{uid[:8]}.png', dpi=100, bbox_inches='tight')
            plt.close()
            viz_count += 1

    # Calculate metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'results.csv', index=False)

    labels = np.array([r['label'] for r in results])

    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")

    for score_name in ['anomaly_score', 'center_score']:
        scores = np.array([r[score_name] for r in results])
        roc_auc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        print(f"{score_name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    scores = np.array([r['anomaly_score'] for r in results])
    print(f"\nHealthy mean score:     {df_results[df_results['label']==0]['anomaly_score'].mean():.4f}")
    print(f"Hemorrhage mean score:  {df_results[df_results['label']==1]['anomaly_score'].mean():.4f}")
    print(f"{'=' * 50}")

    # Save ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc_score(labels, scores):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Patched pDDPM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
