"""
Full Image Diffusion Model Training for CT Hemorrhage Detection
================================================================

Full image training (not patched) - best configuration from sweep:
- 1 channel (brain window)
- 1000 timesteps
- pred_x0 objective
- MSE + SSIM loss
- Full image noising

Usage:
    python train_full.py --output_dir checkpoints_full
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
# CT Preprocessing (1 channel - brain window only)
# =============================================================================

def apply_window(hu_image, window_center, window_width):
    """Apply a single window to HU image, return normalized [0,1]."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(hu_image, img_min, img_max)
    return (windowed - img_min) / (img_max - img_min)


def crop_brain_region(image, threshold=0.1, margin=10):
    """Crop image to brain region, removing black background."""
    # Find rows and columns with content
    row_sums = np.sum(image > threshold, axis=1)
    col_sums = np.sum(image > threshold, axis=0)

    # Find bounding box
    rows_with_content = np.where(row_sums > 0)[0]
    cols_with_content = np.where(col_sums > 0)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image  # Return original if no content found

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

        # Filter to only healthy scans
        pivot = pivot[pivot['any'] == 0]
        self.sop_uids = pivot.index.tolist()

        if num_samples and num_samples < len(self.sop_uids):
            random.seed(seed)
            self.sop_uids = random.sample(self.sop_uids, num_samples)

        # Verify files exist
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
# Model Architecture (UNet for Diffusion)
# =============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bncd,bnce->bnde', q, k) * scale, dim=-1)
        out = torch.einsum('bnde,bnce->bncd', attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """UNet for diffusion model."""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2, dropout=0.1):
        super().__init__()
        time_dim = ch * 4

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(ch),
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        channels = [ch] + [ch * m for m in ch_mult]

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(ch_mult)):
            in_c, out_c = channels[i], channels[i + 1]
            blocks = nn.ModuleList([
                ResBlock(in_c if j == 0 else out_c, out_c, time_dim, dropout)
                for j in range(num_res_blocks)
            ])
            self.down_blocks.append(blocks)
            self.down_samples.append(
                nn.Conv2d(out_c, out_c, 3, stride=2, padding=1) if i < len(ch_mult) - 1 else None
            )

        # Middle
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = Attention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(ch_mult)):
            in_c, skip_c, out_c = rev_channels[i], rev_channels[i + 1], rev_channels[i + 1]
            blocks = nn.ModuleList([
                ResBlock(in_c + skip_c if j == 0 else out_c, out_c, time_dim, dropout)
                for j in range(num_res_blocks)
            ])
            self.up_blocks.append(blocks)
            self.up_samples.append(
                nn.ConvTranspose2d(out_c, out_c, 4, stride=2, padding=1) if i < len(ch_mult) - 1 else None
            )

        self.final = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        skips = [h]
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if downsample:
                h = downsample(h)
        skips = skips[:-1]

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            if upsample:
                h = upsample(h)

        return self.final(h)


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
# Anomaly Detection Evaluation (Full Image)
# =============================================================================

@torch.no_grad()
def one_step_denoise(model, noisy_image, t, scheduler, device):
    """One-step denoising: model directly predicts x0."""
    t_tensor = torch.full((noisy_image.shape[0],), t, device=device, dtype=torch.long)
    output = model(noisy_image, t_tensor)
    return torch.clamp(output, -1, 1)


@torch.no_grad()
def get_anomaly_score(model, image_tensor, scheduler, device, noise_level=150):
    """Compute anomaly score using full image reconstruction."""
    B, C, H, W = image_tensor.shape

    # Add noise to full image
    noise = torch.randn_like(image_tensor)
    t_tensor = torch.full((B,), noise_level, device=device, dtype=torch.long)
    noisy = scheduler.add_noise(image_tensor, noise, t_tensor)

    # One-step denoise
    reconstructed = one_step_denoise(model, noisy, noise_level, scheduler, device)

    # Compute error
    error = torch.abs(image_tensor - reconstructed).mean(dim=1)  # (B, H, W)
    return error.mean().item()


def evaluate_anomaly_detection(model, scheduler, device, data_dir,
                               num_samples=50, seed=42, output_dir=None, epoch=None):
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
    sample_images = {'healthy': [], 'hemorrhage': []}

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
        score = get_anomaly_score(model, image_tensor, scheduler, device)
        scores.append(score)
        labels.append(label)

        key = 'hemorrhage' if label == 1 else 'healthy'
        if len(sample_images[key]) < 3:
            sample_images[key].append((image, score))

    if len(scores) < 10:
        return 0.5, 0.0, 0.0

    scores = np.array(scores)
    labels = np.array(labels)
    roc_auc = roc_auc_score(labels, scores)
    healthy_scores = scores[labels == 0]
    hemorrhage_scores = scores[labels == 1]
    healthy_mean = healthy_scores.mean()
    hemorrhage_mean = hemorrhage_scores.mean()

    # Save detailed diagnostics
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
Epoch {epoch} - FULL IMAGE Training
================================
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
    ax1.plot(epochs, history['roc_auc'], 'b-o', linewidth=2, markersize=6,
             label=f"Full Image (Best: {max(history['roc_auc']):.3f})")
    best_auc = max(history['roc_auc'])
    ax1.axhline(y=best_auc, color='b', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('Anomaly Detection Performance (Full Image Training)')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.35, 0.85)

    ax2 = axes[1]
    ax2.plot(epochs, history['val_loss'], 'b-o', linewidth=2, markersize=6,
             label='Full Image')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training Mode: FULL IMAGE (best from sweep)")

    # Dataset
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

    # Model
    model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.1).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

    for epoch in range(args.epochs):
        # Train - FULL IMAGE
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            B, C, H, W = images.shape
            t = torch.randint(0, scheduler.timesteps, (B,), device=device)

            # Full image noising
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)

            # Predict x0 (clean image)
            pred_x0 = model(noisy, t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # MSE + SSIM loss
            mse_loss = F.mse_loss(pred_x0, images)
            ssim_loss_val = ssim_loss(pred_x0, images)
            loss = mse_loss + ssim_loss_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        noise_level = 150
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                B = images.shape[0]

                noise = torch.randn_like(images)
                t_tensor = torch.full((B,), noise_level, device=device, dtype=torch.long)
                noisy = scheduler.add_noise(images, noise, t_tensor)

                reconstructed = model(noisy, t_tensor)
                reconstructed = torch.clamp(reconstructed, -1, 1)

                error = torch.abs(reconstructed - images)
                val_loss += error.mean().item()

        val_loss /= len(val_loader)

        # Evaluate anomaly detection
        print(f"  Evaluating anomaly detection...")
        roc_auc, healthy_mean, hemorrhage_mean = evaluate_anomaly_detection(
            model, scheduler, device, args.data_dir,
            num_samples=args.eval_samples, seed=args.seed,
            output_dir=output_dir, epoch=epoch+1
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
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'roc_auc': roc_auc,
            }, output_dir / "best_model_loss.pt")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'roc_auc': roc_auc,
            }, output_dir / "best_model.pt")
            print(f"  -> New best ROC-AUC: {roc_auc:.4f}")

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, output_dir)

    print(f"\nDone! Best val loss: {best_val_loss:.6f}, Best ROC-AUC: {best_roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Full Image Diffusion for CT Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_full')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
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
