"""
Patched Diffusion Model (pDDPM) Training for CT Hemorrhage Detection
=====================================================================

Based on: "Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI"
(Behrendt et al., MIDL 2023) - arXiv:2303.03758

Usage:
    # Train from scratch
    python train.py --output_dir checkpoints_scratch

    # Train from pretrained (HuggingFace)
    python train.py --pretrained --output_dir checkpoints_pretrained
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
from sklearn.metrics import roc_auc_score
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
# CT Preprocessing
# =============================================================================

def apply_window(hu_image, window_center, window_width):
    """Apply a single window to HU image, return normalized [0,1]."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(hu_image, img_min, img_max)
    return (windowed - img_min) / (img_max - img_min)


def load_and_preprocess_dicom(filepath: str, image_size: int = 256, num_channels: int = 1) -> Optional[np.ndarray]:
    """Load DICOM and apply windowing.

    Args:
        num_channels: 1 for brain window only, 3 for brain/subdural/bone windows
    """
    try:
        dcm = pydicom.dcmread(filepath)
        pixel_array = dcm.pixel_array.astype(np.float32)

        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        hu_image = pixel_array * slope + intercept

        from PIL import Image

        if num_channels == 1:
            # Brain window only (W:80, L:40) - best from sweep
            brain = apply_window(hu_image, 40, 80)
            img = Image.fromarray((brain * 255).astype(np.uint8))
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            ch = np.array(img).astype(np.float32) / 255.0
            ch = ch * 2.0 - 1.0  # Normalize to [-1, 1]
            return ch[np.newaxis, ...]  # (1, H, W)
        else:
            # 3-channel: brain, subdural, bone windows (for pretrained RGB model)
            brain = apply_window(hu_image, 40, 80)      # Brain window
            subdural = apply_window(hu_image, 80, 200)  # Subdural window
            bone = apply_window(hu_image, 600, 2800)    # Bone window

            channels = []
            for window in [brain, subdural, bone]:
                img = Image.fromarray((window * 255).astype(np.uint8))
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                ch = np.array(img).astype(np.float32) / 255.0
                ch = ch * 2.0 - 1.0  # Normalize to [-1, 1]
                channels.append(ch)
            return np.stack(channels, axis=0)  # (3, H, W)
    except Exception:
        return None


# =============================================================================
# Dataset
# =============================================================================

class CTDataset(Dataset):
    """Dataset for CT scans - loads only healthy scans for training."""

    def __init__(self, data_dir: str, csv_path: str, image_size: int = 256,
                 patch_size: int = 64, num_samples: Optional[int] = None, seed: int = 42,
                 num_channels: int = 1):
        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

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
        image = load_and_preprocess_dicom(str(self.data_dir / f"ID_{uid}.dcm"), self.image_size, self.num_channels)
        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return torch.tensor(image)  # (C, H, W)


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
# HuggingFace Pretrained Model
# =============================================================================

def load_pretrained_model(model_id: str, device: str):
    """Load pretrained UNet from HuggingFace (keep 3 channels for 3-window input)."""
    from diffusers import UNet2DModel

    print(f"Loading pretrained model: {model_id}")
    unet = UNet2DModel.from_pretrained(model_id)

    # Keep 3 channels - our 3-window CT input matches RGB
    print(f"Using {unet.config.in_channels} channels (brain/subdural/bone windows)")

    return unet.to(device)


# =============================================================================
# Anomaly Detection Evaluation
# =============================================================================

@torch.no_grad()
def one_step_denoise(model, noisy_image, t, scheduler, device, is_hf=False):
    """One-step denoising: model directly predicts x0."""
    t_tensor = torch.full((noisy_image.shape[0],), t, device=device, dtype=torch.long)
    output = model(noisy_image, t_tensor)
    x0_pred = output.sample if is_hf and hasattr(output, 'sample') else output
    return torch.clamp(x0_pred, -1, 1)


@torch.no_grad()
def get_anomaly_score(model, image_tensor, scheduler, device, is_hf=False,
                      patch_size=64, stride=32, noise_level=150):
    """Compute anomaly score using patched reconstruction."""
    B, C, H, W = image_tensor.shape
    anomaly_map = torch.zeros((H, W), device=device)
    count_map = torch.zeros((H, W), device=device)

    sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patched_input = image_tensor.clone()
            patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)
            original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            reconstructed = one_step_denoise(model, patched_input, noise_level, scheduler, device, is_hf)
            recon_patch = reconstructed[:, :, y:y+patch_size, x:x+patch_size]
            patch_error = torch.abs(original_patch - recon_patch).mean(dim=1).squeeze()

            anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
            count_map[y:y+patch_size, x:x+patch_size] += 1

    final_map = anomaly_map / torch.clamp(count_map, min=1)
    return final_map.mean().item()


def evaluate_anomaly_detection(model, scheduler, device, data_dir, is_hf=False,
                               num_samples=50, seed=42, output_dir=None, epoch=None,
                               num_channels=1):
    """Evaluate anomaly detection ROC-AUC on a small test set."""
    csv_path = Path(data_dir) / "stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy_uids = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage_uids = pivot[pivot['any'] == 1].index.tolist()

    random.seed(seed + 1000)  # Different seed from training
    test_healthy = random.sample(healthy_uids, min(num_samples, len(healthy_uids)))
    test_hemorrhage = random.sample(hemorrhage_uids, min(num_samples, len(hemorrhage_uids)))

    dicom_dir = Path(data_dir) / "stage_2_train"
    scores, labels = [], []
    sample_images = {'healthy': [], 'hemorrhage': []}  # Store a few for visualization

    model.eval()
    test_samples = [(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage]

    for uid, label in test_samples:
        filepath = dicom_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), 256, num_channels)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).to(device)  # (1, C, H, W)
        score = get_anomaly_score(model, image_tensor, scheduler, device, is_hf)
        scores.append(score)
        labels.append(label)

        # Store a few samples for visualization
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

        # 1. Score distribution histogram
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

        # 2. ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        ax2 = axes[1]
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Score statistics
        ax3 = axes[2]
        ax3.axis('off')
        stats_text = f"""
Epoch {epoch} Evaluation Statistics
================================
Healthy samples:     {len(healthy_scores)}
Hemorrhage samples:  {len(hemorrhage_scores)}

Healthy scores:
  Mean: {healthy_mean:.6f}
  Std:  {healthy_scores.std():.6f}
  Min:  {healthy_scores.min():.6f}
  Max:  {healthy_scores.max():.6f}

Hemorrhage scores:
  Mean: {hemorrhage_mean:.6f}
  Std:  {hemorrhage_scores.std():.6f}
  Min:  {hemorrhage_scores.min():.6f}
  Max:  {hemorrhage_scores.max():.6f}

Separation:
  Difference: {hemorrhage_mean - healthy_mean:.6f}
  ROC-AUC:    {roc_auc:.4f}
"""
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(output_dir / f'eval_scores_epoch_{epoch}.png', dpi=150)
        plt.close()

        # 4. Sample comparisons (healthy vs hemorrhage)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for row, (key, color) in enumerate([('healthy', 'green'), ('hemorrhage', 'red')]):
            for col, (img, score) in enumerate(sample_images[key][:3]):
                ax = axes[row, col]
                # Show brain window (channel 0)
                display_img = (img[0] + 1) / 2
                ax.imshow(display_img, cmap='gray')
                ax.set_title(f'{key.capitalize()}\nScore: {score:.4f}', color=color)
                ax.axis('off')

        plt.suptitle(f'Epoch {epoch}: Sample Images with Anomaly Scores')
        plt.tight_layout()
        plt.savefig(output_dir / f'eval_samples_epoch_{epoch}.png', dpi=150)
        plt.close()

    return roc_auc, healthy_mean, hemorrhage_mean


def plot_training_curves(history, output_dir):
    """Generate comparison plot like the reference image."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    epochs = list(range(1, len(history['roc_auc']) + 1))

    # Top: ROC-AUC
    ax1 = axes[0]
    ax1.plot(epochs, history['roc_auc'], 'g-o', linewidth=2, markersize=6,
             label=f"From Scratch (Best: {max(history['roc_auc']):.3f})")
    best_auc = max(history['roc_auc'])
    ax1.axhline(y=best_auc, color='g', linestyle='--', alpha=0.5)
    ax1.fill_between(epochs, 0.7, best_auc, alpha=0.1, color='g')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('Anomaly Detection Performance')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.35, 0.85)

    # Bottom: Loss
    ax2 = axes[1]
    ax2.plot(epochs, history['val_loss'], 'g-o', linewidth=2, markersize=6,
             label='From Scratch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (MSE)')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training_curves.png")


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine number of channels based on pretrained flag
    num_channels = 3 if args.pretrained else 1
    print(f"Using {num_channels} channel(s): {'brain/subdural/bone windows' if num_channels == 3 else 'brain window only'}")

    # Dataset
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    dataset = CTDataset(args.data_dir, str(csv_path), num_samples=args.num_samples, seed=args.seed, num_channels=num_channels)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model - for 256x256 full images
    if args.pretrained:
        model = load_pretrained_model(args.hf_model, device)
        is_hf = True
    else:
        model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.1).to(device)
        is_hf = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_roc_auc = 0.0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'roc_auc': [],
        'healthy_mean': [],
        'hemorrhage_mean': []
    }

    patch_size = 64

    for epoch in range(args.epochs):
        # Train - PATCHED training (like pDDPM paper)
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

            # Predict x0 (clean image)
            output = model(noisy, t)
            pred_x0 = output.sample if hasattr(output, 'sample') else output
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Loss only on the noised patches (pred_x0 vs original)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate - full image noising at fixed level, measure reconstruction
        model.eval()
        val_loss = 0
        noise_level = 150  # Fixed noise level for evaluation
        save_diagnostics = True  # Save first batch each epoch
        with torch.no_grad():
            for batch_idx, images in enumerate(val_loader):
                images = images.to(device)
                B, C, H, W = images.shape

                # Full image noising at fixed noise level
                noise = torch.randn_like(images)
                t_tensor = torch.full((B,), noise_level, device=device, dtype=torch.long)
                noisy = scheduler.add_noise(images, noise, t_tensor)

                # One-step denoise (model predicts x0 directly)
                output = model(noisy, t_tensor)
                reconstructed = output.sample if hasattr(output, 'sample') else output
                reconstructed = torch.clamp(reconstructed, -1, 1)

                error_map = torch.abs(reconstructed - images)
                val_loss += error_map.mean().item()

                # Save diagnostic images for first batch
                if save_diagnostics and batch_idx == 0:
                    n_show = min(4, B)
                    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
                    if n_show == 1:
                        axes = axes.reshape(1, -1)

                    for i in range(n_show):
                        # Original
                        orig = (images[i, 0].cpu().numpy() + 1) / 2
                        axes[i, 0].imshow(orig, cmap='gray')
                        axes[i, 0].set_title(f'Original')
                        axes[i, 0].axis('off')

                        # Reconstructed
                        recon = (reconstructed[i, 0].cpu().numpy() + 1) / 2
                        axes[i, 1].imshow(recon, cmap='gray')
                        axes[i, 1].set_title(f'Reconstructed')
                        axes[i, 1].axis('off')

                        # Error map
                        err = error_map[i, 0].cpu().numpy()
                        axes[i, 2].imshow(err, cmap='hot')
                        axes[i, 2].set_title(f'Error (mean={err.mean():.4f})')
                        axes[i, 2].axis('off')

                        # Difference (amplified)
                        diff = (recon - orig)
                        axes[i, 3].imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
                        axes[i, 3].set_title(f'Diff (amplified)')
                        axes[i, 3].axis('off')

                    plt.suptitle(f'Epoch {epoch+1} Validation Diagnostics')
                    plt.tight_layout()
                    plt.savefig(output_dir / f'diagnostics_epoch_{epoch+1}.png', dpi=150)
                    plt.close()
                    save_diagnostics = False

        val_loss /= len(val_loader)

        # Evaluate anomaly detection
        print(f"  Evaluating anomaly detection...")
        roc_auc, healthy_mean, hemorrhage_mean = evaluate_anomaly_detection(
            model, scheduler, device, args.data_dir, is_hf,
            num_samples=args.eval_samples, seed=args.seed,
            output_dir=output_dir, epoch=epoch+1,
            num_channels=num_channels
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['roc_auc'].append(roc_auc)
        history['healthy_mean'].append(healthy_mean)
        history['hemorrhage_mean'].append(hemorrhage_mean)

        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, ROC-AUC={roc_auc:.4f}")

        # Save best by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'roc_auc': roc_auc,
                'is_hf': is_hf,
                'hf_model': args.hf_model if is_hf else None,
                'config': dict(model.config) if is_hf else None,
            }, output_dir / "best_model_loss.pt")

        # Save best by ROC-AUC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'roc_auc': roc_auc,
                'is_hf': is_hf,
                'hf_model': args.hf_model if is_hf else None,
                'config': dict(model.config) if is_hf else None,
            }, output_dir / "best_model.pt")
            print(f"  -> New best ROC-AUC: {roc_auc:.4f}")

        # Save history and plot after each epoch
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, output_dir)

    print(f"\nDone! Best val loss: {best_val_loss:.6f}, Best ROC-AUC: {best_roc_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train pDDPM for CT Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_samples', type=int, default=2000, help='Training samples (healthy scans)')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_samples', type=int, default=50, help='Samples per class for eval')
    parser.add_argument('--pretrained', action='store_true', help='Use HuggingFace pretrained model')
    parser.add_argument('--hf_model', type=str, default='google/ddpm-ema-celebahq-256')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
