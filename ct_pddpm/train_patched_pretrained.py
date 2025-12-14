"""
Patched Diffusion Model (pDDPM) Training with Pretrained HuggingFace Weights
=============================================================================

Same as train_patched.py but starts from pretrained HuggingFace weights
to compare with from-scratch training.

Usage:
    python train_patched_pretrained.py --output_dir checkpoints_patched_pretrained
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import random
import json

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# =============================================================================
# SSIM Loss
# =============================================================================

def gaussian_kernel(window_size=11, sigma=1.5, channels=1):
    """Create a Gaussian kernel for SSIM."""
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(img1, img2, window_size=11, sigma=1.5):
    """Compute SSIM loss (1 - SSIM)."""
    channels = img1.shape[1]
    kernel = gaussian_kernel(window_size, sigma, channels).to(img1.device)

    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2, groups=channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


# =============================================================================
# CT Preprocessing (1 channel - brain window only, with brain cropping)
# =============================================================================

def apply_window(hu_image, window_center, window_width):
    """Apply a single window to HU image, return normalized [0,1]."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(hu_image, img_min, img_max)
    return (windowed - img_min) / (img_max - img_min)


def crop_brain_region(image, threshold=0.1, margin=10):
    """Crop image to brain region, removing black background."""
    row_sums = np.sum(image > threshold, axis=1)
    col_sums = np.sum(image > threshold, axis=0)

    rows_with_content = np.where(row_sums > 0)[0]
    cols_with_content = np.where(col_sums > 0)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image

    y_min = max(0, rows_with_content[0] - margin)
    y_max = min(image.shape[0], rows_with_content[-1] + margin)
    x_min = max(0, cols_with_content[0] - margin)
    x_max = min(image.shape[1], cols_with_content[-1] + margin)

    return image[y_min:y_max, x_min:x_max]


def load_and_preprocess_dicom(filepath: str, image_size: int = 256) -> Optional[np.ndarray]:
    """Load DICOM, crop to brain region, and apply brain window only (1 channel)."""
    try:
        dcm = pydicom.dcmread(filepath)
        pixel_array = dcm.pixel_array.astype(np.float32)

        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        hu_image = pixel_array * slope + intercept

        # Brain window only (W:80, L:40)
        brain = apply_window(hu_image, 40, 80)

        # Crop to brain region (remove black background)
        brain_cropped = crop_brain_region(brain, threshold=0.05, margin=5)

        from PIL import Image
        img = Image.fromarray((brain_cropped * 255).astype(np.uint8))
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        ch = np.array(img).astype(np.float32) / 255.0
        ch = ch * 2.0 - 1.0  # Normalize to [-1, 1]

        return ch[np.newaxis, ...]  # (1, H, W)
    except Exception:
        return None


# =============================================================================
# Dataset
# =============================================================================

class CTDataset(Dataset):
    """Dataset for CT scans - loads only healthy scans for training."""

    def __init__(self, data_dir: str, csv_path: str, image_size: int = 256,
                 num_samples: Optional[int] = None, seed: int = 42):
        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        pivot = pivot[pivot['any'] == 0]
        self.sop_uids = pivot.index.tolist()

        if num_samples and num_samples < len(self.sop_uids):
            random.seed(seed)
            self.sop_uids = random.sample(self.sop_uids, num_samples)

        self.valid_uids = []
        for uid in tqdm(self.sop_uids, desc="Verifying files"):
            if (self.data_dir / f"ID_{uid}.dcm").exists():
                self.valid_uids.append(uid)
        self.sop_uids = self.valid_uids
        print(f"Found {len(self.sop_uids)} valid healthy scans")

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        uid = self.sop_uids[idx]
        image = load_and_preprocess_dicom(str(self.data_dir / f"ID_{uid}.dcm"), self.image_size)
        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return torch.tensor(image)


# =============================================================================
# Noise Scheduler
# =============================================================================

class NoiseScheduler:
    def __init__(self, timesteps=1000, schedule='cosine', device='cpu'):
        self.timesteps = timesteps
        self.device = device

        if schedule == 'cosine':
            s = 0.008
            steps = torch.linspace(0, timesteps, timesteps + 1, device=device)
            alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            self.betas = torch.linspace(0.0001, 0.02, timesteps, device=device)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x, noise, t):
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise


# =============================================================================
# HuggingFace UNet Wrapper for pred_x0
# =============================================================================

class HFUNetWrapper(nn.Module):
    """Wrapper to make HuggingFace UNet output pred_x0 instead of pred_noise."""

    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, x, t):
        """Forward pass: predicts x0 by first predicting noise then deriving x0."""
        # HF model predicts noise by default
        output = self.unet(x, t)
        pred_noise = output.sample if hasattr(output, 'sample') else output

        # Derive x0 from predicted noise
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # x0 = (x_t - sqrt(1-alpha) * noise) / sqrt(alpha)
        x0_pred = (x - sqrt_one_minus * pred_noise) / sqrt_alpha
        return torch.clamp(x0_pred, -1, 1)


# =============================================================================
# Anomaly Detection Evaluation (Patched)
# =============================================================================

@torch.no_grad()
def one_step_denoise(model, noisy_image, t, scheduler, device, is_wrapper=False):
    """One-step denoising: model directly predicts x0."""
    t_tensor = torch.full((noisy_image.shape[0],), t, device=device, dtype=torch.long)
    output = model(noisy_image, t_tensor)

    if is_wrapper:
        return output  # Already clamped x0
    return torch.clamp(output, -1, 1)


@torch.no_grad()
def get_anomaly_score(model, image_tensor, scheduler, device,
                      patch_size=64, stride=32, noise_level=150, is_wrapper=False):
    """Compute anomaly score using PATCHED reconstruction."""
    B, C, H, W = image_tensor.shape
    anomaly_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Noise only this patch, keep rest clean
            patched_input = image_tensor.clone()
            patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)
            original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            # Reconstruct
            reconstructed = one_step_denoise(model, patched_input, noise_level, scheduler, device, is_wrapper)
            recon_patch = reconstructed[:, :, y:y+patch_size, x:x+patch_size]
            patch_error = torch.abs(original_patch - recon_patch).mean(dim=1).squeeze()

            anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
            count_map[y:y+patch_size, x:x+patch_size] += 1

    final_map = anomaly_map / torch.clamp(count_map, min=1)
    return final_map.mean().item()


def evaluate_anomaly_detection(model, scheduler, device, data_dir,
                               num_samples=50, seed=42, output_dir=None, epoch=None, is_wrapper=False):
    """Evaluate anomaly detection ROC-AUC on a small test set."""
    csv_path = Path(data_dir) / "stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy_uids = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage_uids = pivot[pivot['any'] == 1].index.tolist()

    random.seed(seed + 1000)
    test_healthy = random.sample(healthy_uids, min(num_samples, len(healthy_uids)))
    test_hemorrhage = random.sample(hemorrhage_uids, min(num_samples, len(hemorrhage_uids)))

    dicom_dir = Path(data_dir) / "stage_2_train"
    scores, labels = [], []

    model.eval()
    test_samples = [(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage]

    for uid, label in test_samples:
        filepath = dicom_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), 256)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).to(device)
        score = get_anomaly_score(model, image_tensor, scheduler, device, is_wrapper=is_wrapper)
        scores.append(score)
        labels.append(label)

    if len(scores) < 10:
        return 0.5, 0.0, 0.0

    scores = np.array(scores)
    labels = np.array(labels)
    roc_auc = roc_auc_score(labels, scores)
    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]
    healthy_mean = healthy_scores.mean()
    hemorrhage_mean = hemorrhage_scores.mean()

    if output_dir is not None and epoch is not None:
        output_dir = Path(output_dir)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax1 = axes[0]
        ax1.hist(healthy_scores, bins=20, alpha=0.7, label=f'Healthy (n={len(healthy_scores)})', color='green')
        ax1.hist(hemorrhage_scores, bins=20, alpha=0.7, label=f'Hemorrhage (n={len(hemorrhage_scores)})', color='red')
        ax1.axvline(healthy_mean, color='green', linestyle='--', label=f'Healthy mean: {healthy_mean:.4f}')
        ax1.axvline(hemorrhage_mean, color='red', linestyle='--', label=f'Hemorrhage mean: {hemorrhage_mean:.4f}')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Score Distribution (ROC-AUC: {roc_auc:.4f})')
        ax1.legend()

        fpr, tpr, _ = roc_curve(labels, scores)
        ax2 = axes[1]
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(alpha=0.3)

        ax3 = axes[2]
        ax3.axis('off')
        stats_text = f"""
Epoch {epoch} - PRETRAINED PATCHED Training
============================================
Healthy:     mean={healthy_mean:.6f}, std={healthy_scores.std():.6f}
Hemorrhage:  mean={hemorrhage_mean:.6f}, std={hemorrhage_scores.std():.6f}
Separation:  {hemorrhage_mean - healthy_mean:.6f}
ROC-AUC:     {roc_auc:.4f}
"""
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(output_dir / f'eval_epoch_{epoch}.png', dpi=150)
        plt.close()

    return roc_auc, healthy_mean, hemorrhage_mean


def plot_training_curves(history, output_dir):
    """Generate training curves plot."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    epochs = list(range(1, len(history['roc_auc']) + 1))

    ax1 = axes[0]
    ax1.plot(epochs, history['roc_auc'], 'r-s', linewidth=2, markersize=6,
             label=f"Pretrained Patched (Best: {max(history['roc_auc']):.3f})")
    best_auc = max(history['roc_auc'])
    ax1.axhline(y=best_auc, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('Anomaly Detection Performance (Pretrained Patched Training)')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.35, 0.85)

    ax2 = axes[1]
    ax2.plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=6,
             label='Pretrained Patched')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Training with Pretrained HuggingFace Weights
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training Mode: PATCHED with PRETRAINED HuggingFace weights")

    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    dataset = CTDataset(args.data_dir, str(csv_path), num_samples=args.num_samples, seed=args.seed)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Load pretrained HuggingFace model
    from diffusers import UNet2DModel

    hf_model_id = "google/ddpm-ema-celebahq-256"
    print(f"Loading pretrained model: {hf_model_id}")

    base_model = UNet2DModel.from_pretrained(hf_model_id)
    config = dict(base_model.config)

    # Modify for 1 channel grayscale
    config['in_channels'] = 1
    config['out_channels'] = 1

    unet = UNet2DModel(**config).to(device)

    # Initialize first/last conv layers for 1 channel
    # Copy center channel weights from RGB pretrained model
    with torch.no_grad():
        # Input conv: take mean of RGB weights
        pretrained_conv_in = base_model.conv_in.weight.data
        unet.conv_in.weight.data = pretrained_conv_in.mean(dim=1, keepdim=True)
        unet.conv_in.bias.data = base_model.conv_in.bias.data.clone()

        # Output conv: take mean of RGB weights
        pretrained_conv_out = base_model.conv_out.weight.data
        unet.conv_out.weight.data = pretrained_conv_out.mean(dim=0, keepdim=True)
        unet.conv_out.bias.data = base_model.conv_out.bias.data[:1].clone()

    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)

    # Wrap model to output pred_x0
    model = HFUNetWrapper(unet, scheduler).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_roc_auc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'roc_auc': [],
        'healthy_mean': [],
        'hemorrhage_mean': []
    }

    patch_size = 64

    for epoch in range(args.epochs):
        # Train - PATCHED
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            B, C, H, W = images.shape
            t = torch.randint(0, scheduler.timesteps, (B,), device=device)

            # Patched noising: only noise a random patch, keep rest clean
            noisy = images.clone()
            patch_coords = []
            for i in range(B):
                y = random.randint(0, H - patch_size)
                x = random.randint(0, W - patch_size)
                patch_coords.append((y, x))
                patch_noise = torch.randn(C, patch_size, patch_size, device=device)
                noisy[i, :, y:y+patch_size, x:x+patch_size] = scheduler.add_noise(
                    images[i:i+1, :, y:y+patch_size, x:x+patch_size],
                    patch_noise.unsqueeze(0),
                    t[i:i+1]
                ).squeeze(0)

            # Predict x0 (clean image) via wrapper
            pred_x0 = model(noisy, t)

            # Loss only on the noised patches
            mse_loss = 0
            ssim_loss_val = 0
            for i, (y, x) in enumerate(patch_coords):
                pred_patch = pred_x0[i:i+1, :, y:y+patch_size, x:x+patch_size]
                orig_patch = images[i:i+1, :, y:y+patch_size, x:x+patch_size]
                mse_loss += F.mse_loss(pred_patch, orig_patch)
                ssim_loss_val += ssim_loss(pred_patch, orig_patch)

            mse_loss /= B
            ssim_loss_val /= B
            loss = mse_loss + ssim_loss_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate - PATCHED (slide through patches)
        model.eval()
        val_loss = 0
        noise_level = 150
        stride = 64  # Non-overlapping for validation
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                B, C, H, W = images.shape

                reconstructed = torch.zeros_like(images)
                count_map = torch.zeros((B, 1, H, W), device=device)

                sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
                sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        patched_input = images.clone()
                        patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)
                        original_patch = images[:, :, y:y+patch_size, x:x+patch_size]
                        noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
                        patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

                        t_tensor = torch.full((B,), noise_level, device=device, dtype=torch.long)
                        x0_pred = model(patched_input, t_tensor)

                        reconstructed[:, :, y:y+patch_size, x:x+patch_size] += x0_pred[:, :, y:y+patch_size, x:x+patch_size]
                        count_map[:, :, y:y+patch_size, x:x+patch_size] += 1

                reconstructed = reconstructed / torch.clamp(count_map, min=1)
                error = torch.abs(reconstructed - images)
                val_loss += error.mean().item()

        val_loss /= len(val_loader)

        print(f"  Evaluating anomaly detection...")
        roc_auc, healthy_mean, hemorrhage_mean = evaluate_anomaly_detection(
            model, scheduler, device, args.data_dir,
            num_samples=args.eval_samples, seed=args.seed,
            output_dir=output_dir, epoch=epoch+1, is_wrapper=True
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['roc_auc'].append(roc_auc)
        history['healthy_mean'].append(healthy_mean)
        history['hemorrhage_mean'].append(hemorrhage_mean)

        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, ROC-AUC={roc_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': {'unet.' + k: v for k, v in unet.state_dict().items()},
                'val_loss': val_loss,
                'roc_auc': roc_auc,
                'is_hf': True,
                'hf_model': hf_model_id,
            }, output_dir / "best_model_loss.pt")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': {'unet.' + k: v for k, v in unet.state_dict().items()},
                'val_loss': val_loss,
                'roc_auc': roc_auc,
                'is_hf': True,
                'hf_model': hf_model_id,
            }, output_dir / "best_model.pt")
            print(f"  -> New best ROC-AUC: {roc_auc:.4f}")

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, output_dir)

    print(f"\nDone! Best val loss: {best_val_loss:.6f}, Best ROC-AUC: {best_roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Pretrained Patched Diffusion for CT Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_patched_pretrained')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
