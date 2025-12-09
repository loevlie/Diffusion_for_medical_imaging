"""
Patched DDPM Inference for Anomaly Detection
=============================================

This script implements the pDDPM inference procedure for detecting hemorrhages
in brain CT scans.

Key steps:
1. Load trained DDPM model (trained on healthy scans only)
2. For each test image, divide into overlapping patches
3. Add noise to each patch while keeping context clean
4. Denoise each patch using the trained model
5. Stitch patches together
6. Compute anomaly score as difference between original and reconstruction

Reference: Behrendt et al., "Patched Diffusion Models for Unsupervised
Anomaly Detection in Brain MRI", MIDL 2023

Usage:
    python inference_pddpm.py --checkpoint ./checkpoints/best_model.pt --test_image /path/to/image.dcm
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Import from training script
from train_pddpm import (
    UNet, NoiseScheduler, load_and_preprocess_dicom,
    apply_window, CTDataset
)


# =============================================================================
# pDDPM Anomaly Detection
# =============================================================================

class PatchedDDPMAnomaly:
    """Patched DDPM for anomaly detection.

    Key parameters from the paper:
    - Patch size: 60x60 (optimal)
    - Test noise level: t=400 (optimal)
    - Stride: patch_size // 2 for overlap
    """

    def __init__(self,
                 model: UNet,
                 scheduler: NoiseScheduler,
                 device: torch.device,
                 patch_size: int = 60,
                 test_noise_level: int = 400,
                 stride: Optional[int] = None):

        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.patch_size = patch_size
        self.test_noise_level = test_noise_level
        self.stride = stride if stride else patch_size // 2

    def get_patch_positions(self, image_size: int) -> List[Tuple[int, int]]:
        """Get all patch positions for tiled reconstruction."""
        positions = []
        for y in range(0, image_size - self.patch_size + 1, self.stride):
            for x in range(0, image_size - self.patch_size + 1, self.stride):
                positions.append((y, x))

        # Add edge positions if needed
        if positions[-1][0] + self.patch_size < image_size:
            for x in range(0, image_size - self.patch_size + 1, self.stride):
                positions.append((image_size - self.patch_size, x))
        if positions[-1][1] + self.patch_size < image_size:
            for y in range(0, image_size - self.patch_size + 1, self.stride):
                positions.append((y, image_size - self.patch_size))

        return positions

    def add_patch_noise(self,
                        image: torch.Tensor,
                        patch_pos: Tuple[int, int],
                        t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise only to the patch region, keeping context clean.

        Args:
            image: Clean image [B, C, H, W]
            patch_pos: (y, x) position of patch
            t: Noise timestep

        Returns:
            noised_image: Image with noise added to patch
            mask: Binary mask of patch region
        """
        B, C, H, W = image.shape
        y, x = patch_pos

        # Create mask (1 = patch, 0 = context)
        mask = torch.zeros_like(image)
        mask[:, :, y:y+self.patch_size, x:x+self.patch_size] = 1.0

        # Generate noise
        noise = torch.randn_like(image)

        # Add noise only to patch region
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_tensor][:, None, None, None]
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t_tensor][:, None, None, None]

        # Noised patch
        noised_patch = sqrt_alpha * image + sqrt_one_minus * noise

        # Combine: noised patch + clean context
        noised_image = mask * noised_patch + (1 - mask) * image

        return noised_image, mask

    @torch.no_grad()
    def denoise_patch(self,
                      noised_image: torch.Tensor,
                      mask: torch.Tensor,
                      start_t: int) -> torch.Tensor:
        """Denoise the patch region while preserving context.

        The context acts as conditioning information to help reconstruct
        the patch region.
        """
        self.model.eval()
        x = noised_image.clone()

        # Denoise from start_t down to 0
        for t in range(start_t, -1, -1):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            pred_noise = self.model(x, t_tensor)

            # Only update the patch region
            if t > 0:
                alpha = self.scheduler.alphas[t]
                alpha_cumprod = self.scheduler.alphas_cumprod[t]
                beta = self.scheduler.betas[t]

                # Predict x_0
                x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                # Compute mean
                mean = (torch.sqrt(self.scheduler.alphas_cumprod_prev[t]) * beta / (1 - alpha_cumprod)) * x0_pred + \
                       (torch.sqrt(alpha) * (1 - self.scheduler.alphas_cumprod_prev[t]) / (1 - alpha_cumprod)) * x

                # Add noise
                noise = torch.randn_like(x)
                variance = torch.sqrt(self.scheduler.posterior_variance[t])
                x_new = mean + variance * noise

                # Only update patch region, keep context clean
                x = mask * x_new + (1 - mask) * noised_image
            else:
                # Final step: predict x_0
                alpha_cumprod = self.scheduler.alphas_cumprod[0]
                x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
                x0_pred = torch.clamp(x0_pred, -1, 1)
                x = mask * x0_pred + (1 - mask) * noised_image

        return x

    def reconstruct_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full image using patched approach.

        Args:
            image: Input image [1, 1, H, W]

        Returns:
            reconstruction: Reconstructed healthy image
            anomaly_map: Pixel-wise anomaly scores
        """
        _, _, H, W = image.shape
        positions = self.get_patch_positions(H)

        # Accumulators for averaging overlapping regions
        recon_sum = torch.zeros_like(image)
        count = torch.zeros_like(image)

        for y, x in tqdm(positions, desc="Processing patches", leave=False):
            # Create mask for this patch
            mask = torch.zeros_like(image)
            mask[:, :, y:y+self.patch_size, x:x+self.patch_size] = 1.0

            # Add noise to patch
            noised_image, _ = self.add_patch_noise(image, (y, x), self.test_noise_level)

            # Denoise
            recon = self.denoise_patch(noised_image, mask, self.test_noise_level)

            # Accumulate reconstruction for patch region
            recon_sum[:, :, y:y+self.patch_size, x:x+self.patch_size] += \
                recon[:, :, y:y+self.patch_size, x:x+self.patch_size]
            count[:, :, y:y+self.patch_size, x:x+self.patch_size] += 1

        # Average overlapping regions
        reconstruction = recon_sum / count.clamp(min=1)

        # Compute anomaly map (absolute difference)
        anomaly_map = torch.abs(image - reconstruction)

        return reconstruction, anomaly_map

    def detect_anomaly(self,
                       image: torch.Tensor,
                       threshold: Optional[float] = None) -> Dict:
        """Full anomaly detection pipeline.

        Args:
            image: Input image [1, 1, H, W]
            threshold: Optional threshold for binary anomaly mask

        Returns:
            Dictionary with reconstruction, anomaly_map, score, and mask
        """
        reconstruction, anomaly_map = self.reconstruct_image(image)

        # Global anomaly score (mean of anomaly map)
        anomaly_score = anomaly_map.mean().item()

        # Optional binary mask
        binary_mask = None
        if threshold is not None:
            binary_mask = (anomaly_map > threshold).float()

        return {
            'reconstruction': reconstruction,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score,
            'binary_mask': binary_mask
        }


# =============================================================================
# Evaluation Dataset (includes hemorrhage scans)
# =============================================================================

class CTEvalDataset(Dataset):
    """Dataset for evaluation - includes both healthy and hemorrhage scans."""

    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 image_size: int = 256,
                 num_healthy: int = 100,
                 num_hemorrhage: int = 100,
                 seed: int = 42):

        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size

        # Parse labels
        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        # Split into healthy and hemorrhage
        healthy = pivot[pivot['any'] == 0].index.tolist()
        hemorrhage = pivot[pivot['any'] == 1].index.tolist()

        random.seed(seed)
        healthy_sample = random.sample(healthy, min(num_healthy, len(healthy)))
        hemorrhage_sample = random.sample(hemorrhage, min(num_hemorrhage, len(hemorrhage)))

        self.samples = []
        for uid in healthy_sample:
            filepath = self.data_dir / f"ID_{uid}.dcm"
            if filepath.exists():
                self.samples.append((uid, 0))  # 0 = healthy

        for uid in hemorrhage_sample:
            filepath = self.data_dir / f"ID_{uid}.dcm"
            if filepath.exists():
                self.samples.append((uid, 1))  # 1 = hemorrhage

        print(f"Evaluation dataset: {len([s for s in self.samples if s[1]==0])} healthy, "
              f"{len([s for s in self.samples if s[1]==1])} hemorrhage")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, label = self.samples[idx]
        filepath = self.data_dir / f"ID_{uid}.dcm"

        image = load_and_preprocess_dicom(str(filepath), self.image_size)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = torch.tensor(image).unsqueeze(0)
        return image, label, uid


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = UNet(
        in_ch=1,
        out_ch=1,
        ch=args.base_channels,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(32, 16, 8),
        dropout=0.0  # No dropout at inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create scheduler
    scheduler = NoiseScheduler(
        timesteps=args.timesteps,
        schedule='cosine',
        device=device
    )

    # Create anomaly detector
    detector = PatchedDDPMAnomaly(
        model=model,
        scheduler=scheduler,
        device=device,
        patch_size=args.patch_size,
        test_noise_level=args.noise_level,
        stride=args.stride
    )

    # Load evaluation dataset
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    eval_dataset = CTEvalDataset(
        data_dir=args.data_dir,
        csv_path=str(csv_path),
        image_size=args.image_size,
        num_healthy=args.num_healthy,
        num_hemorrhage=args.num_hemorrhage,
        seed=args.seed
    )

    # Evaluate
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (image, label, uid) in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        image = image.unsqueeze(0).to(device)  # [1, 1, H, W]

        # Detect anomaly
        result = detector.detect_anomaly(image)

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': result['anomaly_score']
        })

        # Save visualizations for first few samples
        if i < args.num_visualize:
            save_visualization(
                image=image,
                reconstruction=result['reconstruction'],
                anomaly_map=result['anomaly_map'],
                label=label,
                uid=uid,
                output_dir=output_dir
            )

    # Compute metrics
    results_df = pd.DataFrame(results)
    compute_metrics(results_df, output_dir)

    return results_df


def save_visualization(image, reconstruction, anomaly_map, label, uid, output_dir):
    """Save visualization of detection results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original
    img_np = ((image[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f"Original ({'Hemorrhage' if label else 'Healthy'})")
    axes[0].axis('off')

    # Reconstruction
    recon_np = ((reconstruction[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    # Difference
    diff_np = np.abs(img_np.astype(float) - recon_np.astype(float))
    axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title("Difference")
    axes[2].axis('off')

    # Anomaly map (normalized)
    amap_np = anomaly_map[0, 0].cpu().numpy()
    axes[3].imshow(amap_np, cmap='jet')
    axes[3].set_title(f"Anomaly Map (score: {amap_np.mean():.4f})")
    axes[3].axis('off')

    plt.tight_layout()
    label_str = "hemorrhage" if label else "healthy"
    plt.savefig(output_dir / f"{uid}_{label_str}.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(results_df: pd.DataFrame, output_dir: Path):
    """Compute and save evaluation metrics."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    labels = results_df['label'].values
    scores = results_df['anomaly_score'].values

    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"{'='*50}")

    # Distribution of scores
    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]

    print(f"\nHealthy scores:     mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}")
    print(f"Hemorrhage scores:  mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}")

    # Plot score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(healthy_scores, bins=30, alpha=0.7, label='Healthy', color='green')
    axes[0].hist(hemorrhage_scores, bins=30, alpha=0.7, label='Hemorrhage', color='red')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Score Distribution')
    axes[0].legend()

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[1].plot(fpr, tpr, color='blue', label=f'ROC (AUC={roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_df.to_csv(output_dir / "results.csv", index=False)

    with open(output_dir / "metrics.txt", 'w') as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Healthy scores: mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}\n")
        f.write(f"Hemorrhage scores: mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}\n")


def process_single_image(args):
    """Process a single image for anomaly detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(
        in_ch=1, out_ch=1, ch=args.base_channels,
        ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        attn_resolutions=(32, 16, 8), dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scheduler = NoiseScheduler(timesteps=args.timesteps, schedule='cosine', device=device)

    detector = PatchedDDPMAnomaly(
        model=model, scheduler=scheduler, device=device,
        patch_size=args.patch_size, test_noise_level=args.noise_level
    )

    # Load image
    image = load_and_preprocess_dicom(args.test_image, args.image_size)
    if image is None:
        print(f"Failed to load {args.test_image}")
        return

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    # Detect
    result = detector.detect_anomaly(image)

    print(f"Anomaly Score: {result['anomaly_score']:.4f}")

    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(
        image=image,
        reconstruction=result['reconstruction'],
        anomaly_map=result['anomaly_map'],
        label=-1,  # Unknown
        uid=Path(args.test_image).stem,
        output_dir=output_dir
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='pDDPM Inference for Anomaly Detection')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data',
                        help='Path to RSNA dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to single DICOM file for testing')

    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=256)

    # pDDPM parameters (from paper)
    parser.add_argument('--patch_size', type=int, default=60,
                        help='Patch size (paper: 60)')
    parser.add_argument('--noise_level', type=int, default=400,
                        help='Test noise level t (paper: 400)')
    parser.add_argument('--stride', type=int, default=30,
                        help='Stride for patch extraction')

    # Evaluation
    parser.add_argument('--num_healthy', type=int, default=100,
                        help='Number of healthy samples for evaluation')
    parser.add_argument('--num_hemorrhage', type=int, default=100,
                        help='Number of hemorrhage samples for evaluation')
    parser.add_argument('--num_visualize', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.test_image:
        process_single_image(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
