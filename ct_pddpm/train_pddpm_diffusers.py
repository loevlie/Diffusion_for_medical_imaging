"""
Patched Diffusion Model (pDDPM) for CT Anomaly Detection - Diffusers Version
=============================================================================

Simplified implementation using Hugging Face Diffusers library.
This version is easier to use and leverages well-tested components.

Based on: "Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI"
(Behrendt et al., MIDL 2023) - arXiv:2303.03758

Usage:
    # Train on healthy CT scans
    python train_pddpm_diffusers.py --num_samples 5000 --epochs 50

    # Quick test with fewer samples
    python train_pddpm_diffusers.py --num_samples 1000 --epochs 20 --batch_size 8
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import random

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# CT Preprocessing
# =============================================================================

def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """Apply CT windowing to convert HU values to displayable range."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed.astype(np.float32)


def load_and_preprocess_dicom(filepath: str, image_size: int = 256) -> Optional[np.ndarray]:
    """Load DICOM and apply brain window preprocessing."""
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
        return None


# =============================================================================
# Dataset
# =============================================================================

class CTDataset(Dataset):
    """CT Dataset for training - healthy scans only."""

    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 healthy_only: bool = True,
                 image_size: int = 256,
                 num_samples: Optional[int] = None,
                 seed: int = 42):

        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        if healthy_only:
            pivot = pivot[pivot['any'] == 0]

        self.sop_uids = pivot.index.tolist()

        if num_samples and num_samples < len(self.sop_uids):
            random.seed(seed)
            self.sop_uids = random.sample(self.sop_uids, num_samples)

        # Filter to existing files
        valid_uids = []
        for uid in tqdm(self.sop_uids, desc="Checking files"):
            if (self.data_dir / f"ID_{uid}.dcm").exists():
                valid_uids.append(uid)
        self.sop_uids = valid_uids

        print(f"Dataset: {len(self.sop_uids)} {'healthy' if healthy_only else 'total'} scans")

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        uid = self.sop_uids[idx]
        filepath = self.data_dir / f"ID_{uid}.dcm"

        image = load_and_preprocess_dicom(str(filepath), self.image_size)
        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return torch.tensor(image).unsqueeze(0)


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    dataset = CTDataset(
        data_dir=args.data_dir,
        csv_path=str(csv_path),
        healthy_only=True,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model using diffusers
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.timesteps,
        beta_schedule="squaredcos_cap_v2"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images in pbar:
            images = images.to(device)
            batch_size = images.shape[0]

            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()

            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'image_size': args.image_size,
                    'timesteps': args.timesteps,
                }
            }, output_dir / "best_model.pt")

        # Periodic saves
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

            generate_samples(model, scheduler, device, args.image_size,
                           output_dir / f"samples_epoch_{epoch+1}.png")

    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")


def generate_samples(model, scheduler, device, image_size, output_path, num_samples=16):
    """Generate samples from trained model."""
    model.eval()
    scheduler.set_timesteps(1000)

    with torch.no_grad():
        samples = torch.randn(num_samples, 1, image_size, image_size, device=device)

        for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
            residual = model(samples, t).sample
            samples = scheduler.step(residual, t, samples).prev_sample

        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1).cpu()

    # Plot
    nrow = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(nrow, nrow, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved samples to {output_path}")


# =============================================================================
# Anomaly Detection (pDDPM Inference)
# =============================================================================

def detect_anomalies(args):
    """Run anomaly detection on test samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {'image_size': 256, 'timesteps': 1000})

    model = UNet2DModel(
        sample_size=config['image_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=config['timesteps'],
        beta_schedule="squaredcos_cap_v2"
    )

    # Load test data
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    # Sample healthy and hemorrhage
    healthy = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage = pivot[pivot['any'] == 1].index.tolist()

    random.seed(args.seed)
    test_healthy = random.sample(healthy, min(args.num_test, len(healthy)))
    test_hemorrhage = random.sample(hemorrhage, min(args.num_test, len(hemorrhage)))

    data_dir = Path(args.data_dir) / "stage_2_train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for uid, label in tqdm([(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage],
                           desc="Processing"):
        filepath = data_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), config['image_size'])
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

        # pDDPM: Add noise then denoise
        # Using t=400 as per paper
        t = torch.tensor([args.noise_level], device=device).long()

        noise = torch.randn_like(image_tensor)
        noisy = scheduler.add_noise(image_tensor, noise, t)

        # Denoise
        scheduler.set_timesteps(config['timesteps'])
        sample = noisy.clone()

        # Only denoise from noise_level
        relevant_timesteps = [ts for ts in scheduler.timesteps if ts <= args.noise_level]

        with torch.no_grad():
            for ts in relevant_timesteps:
                residual = model(sample, ts).sample
                sample = scheduler.step(residual, ts, sample).prev_sample

        reconstruction = sample

        # Anomaly score
        anomaly_map = torch.abs(image_tensor - reconstruction)
        anomaly_score = anomaly_map.mean().item()

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_score
        })

        # Visualize first few
        if len(results) <= args.num_visualize:
            visualize_result(
                image_tensor, reconstruction, anomaly_map,
                label, uid, anomaly_score, output_dir
            )

    # Compute metrics
    results_df = pd.DataFrame(results)
    compute_metrics(results_df, output_dir)


def visualize_result(image, reconstruction, anomaly_map, label, uid, score, output_dir):
    """Save visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img_np = ((image[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f"Original ({'Hemorrhage' if label else 'Healthy'})")
    axes[0].axis('off')

    recon_np = ((reconstruction[0, 0].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    diff_np = np.abs(img_np.astype(float) - recon_np.astype(float))
    axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title("Difference")
    axes[2].axis('off')

    amap_np = anomaly_map[0, 0].cpu().numpy()
    axes[3].imshow(amap_np, cmap='jet')
    axes[3].set_title(f"Anomaly (score: {score:.4f})")
    axes[3].axis('off')

    plt.tight_layout()
    label_str = "hemorrhage" if label else "healthy"
    plt.savefig(output_dir / f"{uid}_{label_str}.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(results_df, output_dir):
    """Compute evaluation metrics."""
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
    except ImportError:
        print("sklearn not found, skipping metric computation")
        results_df.to_csv(output_dir / "results.csv", index=False)
        return

    labels = results_df['label'].values
    scores = results_df['anomaly_score'].values

    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*50}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"{'='*50}")

    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]
    print(f"Healthy:     mean={healthy_scores.mean():.4f}, std={healthy_scores.std():.4f}")
    print(f"Hemorrhage:  mean={hemorrhage_scores.mean():.4f}, std={hemorrhage_scores.std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(healthy_scores, bins=30, alpha=0.7, label='Healthy', color='green')
    axes[0].hist(hemorrhage_scores, bins=30, alpha=0.7, label='Hemorrhage', color='red')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title('Score Distribution')

    fpr, tpr, _ = roc_curve(labels, scores)
    axes[1].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].legend()
    axes[1].set_title('ROC Curve')

    plt.tight_layout()
    plt.savefig(output_dir / "metrics.png", dpi=150)
    plt.close()

    results_df.to_csv(output_dir / "results.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='pDDPM for CT Anomaly Detection (Diffusers)')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'detect'],
                        help='Mode: train or detect')

    # Data
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_diffusers')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=256)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # Detection
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_diffusers/best_model.pt')
    parser.add_argument('--noise_level', type=int, default=400, help='pDDPM noise level (paper: 400)')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--num_visualize', type=int, default=20)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'train':
        train(args)
    else:
        detect_anomalies(args)


if __name__ == '__main__':
    main()
