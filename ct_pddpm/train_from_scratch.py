"""
Patched Diffusion Model (pDDPM) - Training FROM SCRATCH
=======================================================

Based on: "Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI"
(Behrendt et al., MIDL 2023) - arXiv:2303.03758

This version trains a DDPM from scratch on CT brain scans.

Usage:
    python train_from_scratch.py --data_dir /media/M2SSD/gen_models_data --num_samples 10000
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# CT Preprocessing (adapted from rsna_ct_preprocessing.py)
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

        # Get rescale parameters for HU conversion
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        slope = float(getattr(dcm, 'RescaleSlope', 1))

        # Convert to Hounsfield Units
        hu_image = pixel_array * slope + intercept

        # Apply brain window (W:80, L:40) - optimal for soft tissue/hemorrhage
        windowed = apply_window(hu_image, window_center=40, window_width=80)

        # Resize to target size
        from PIL import Image
        img = Image.fromarray((windowed * 255).astype(np.uint8))
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

        # Normalize to [-1, 1] for diffusion model
        result = np.array(img).astype(np.float32) / 255.0
        result = result * 2.0 - 1.0

        return result

    except Exception as e:
        return None


# =============================================================================
# Dataset
# =============================================================================

class CTDataset(Dataset):
    """Dataset for CT scans - loads only healthy (non-hemorrhage) scans for training."""

    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 healthy_only: bool = True,
                 image_size: int = 256,
                 num_samples: Optional[int] = None,
                 seed: int = 42):

        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size

        # Parse labels
        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])

        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        if healthy_only:
            # Filter to only healthy scans (no hemorrhage)
            pivot = pivot[pivot['any'] == 0]
            print(f"Filtered to {len(pivot)} healthy scans")

        self.sop_uids = pivot.index.tolist()

        # Subsample if requested
        if num_samples and num_samples < len(self.sop_uids):
            random.seed(seed)
            self.sop_uids = random.sample(self.sop_uids, num_samples)
            print(f"Subsampled to {len(self.sop_uids)} scans")

        # Verify files exist
        self.valid_uids = []
        for uid in tqdm(self.sop_uids[:min(len(self.sop_uids), 100000)], desc="Verifying files"):
            filepath = self.data_dir / f"ID_{uid}.dcm"
            if filepath.exists():
                self.valid_uids.append(uid)

        print(f"Found {len(self.valid_uids)} valid DICOM files")
        self.sop_uids = self.valid_uids

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        uid = self.sop_uids[idx]
        filepath = self.data_dir / f"ID_{uid}.dcm"

        image = load_and_preprocess_dicom(str(filepath), self.image_size)

        if image is None:
            # Return a random other sample if this one fails
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Add channel dimension [H, W] -> [1, H, W]
        image = torch.tensor(image).unsqueeze(0)

        return image


# =============================================================================
# Model Architecture (UNet for Diffusion)
# =============================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time step embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
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
    """Self-attention block."""
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

        # Scaled dot-product attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bncd,bnce->bnde', q, k) * scale, dim=-1)
        out = torch.einsum('bnde,bnce->bncd', attn, v)
        out = out.reshape(b, c, h, w)

        return x + self.proj(out)


class UNet(nn.Module):
    """UNet architecture for diffusion model.

    Simplified and robust implementation with proper channel tracking.
    """

    def __init__(self,
                 in_ch=1,
                 out_ch=1,
                 ch=64,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(32, 16, 8),
                 dropout=0.1):
        super().__init__()

        self.in_ch = in_ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        time_dim = ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(ch),
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Build channel schedule
        channels = [ch]
        for mult in ch_mult:
            channels.append(ch * mult)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for i in range(len(ch_mult)):
            in_c = channels[i]
            out_c = channels[i + 1]

            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(in_c if j == 0 else out_c, out_c, time_dim, dropout))

            self.down_blocks.append(blocks)

            # Downsample except for last level
            if i < len(ch_mult) - 1:
                self.down_samples.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1))
            else:
                self.down_samples.append(None)

        # Middle blocks
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = Attention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        rev_channels = list(reversed(channels))
        for i in range(len(ch_mult)):
            in_c = rev_channels[i]  # From previous level
            skip_c = rev_channels[i + 1]  # Skip connection channel
            out_c = rev_channels[i + 1]  # Output channel

            blocks = nn.ModuleList()
            # First block takes concatenated input
            blocks.append(ResBlock(in_c + skip_c, out_c, time_dim, dropout))
            # Rest of blocks
            for j in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_c, out_c, time_dim, dropout))

            self.up_blocks.append(blocks)

            # Upsample except for last level
            if i < len(ch_mult) - 1:
                self.up_samples.append(nn.ConvTranspose2d(out_c, out_c, 4, stride=2, padding=1))
            else:
                self.up_samples.append(None)

        # Output
        self.final = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        # Downsampling with skip connections
        skips = [h]
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if downsample is not None:
                h = downsample(h)

        # Remove last skip (it's just h before middle blocks)
        skips = skips[:-1]

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling with skip connections
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()

            # Handle size mismatch
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='nearest')

            # Concatenate skip connection
            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = block(h, t_emb)

            if upsample is not None:
                h = upsample(h)

        return self.final(h)


# =============================================================================
# Noise Scheduler
# =============================================================================

class NoiseScheduler:
    """DDPM noise scheduler with linear or cosine schedule."""

    def __init__(self,
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 schedule='linear',
                 device='cpu'):
        self.timesteps = timesteps
        self.device = device

        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule == 'cosine':
            # Cosine schedule from "Improved DDPM"
            s = 0.008
            steps = torch.linspace(0, timesteps, timesteps + 1, device=device)
            alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(betas, 0.0001, 0.999)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def add_noise(self, x, noise, t):
        """Forward diffusion: add noise to x at timestep t."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise

    @torch.no_grad()
    def sample_step(self, model, x, t, clip_denoised=True):
        """Reverse diffusion: denoise x from timestep t to t-1."""
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = model(x, t_tensor)

        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1, 1)

        # Compute mean
        mean = (torch.sqrt(self.alphas_cumprod_prev[t]) * beta / (1 - alpha_cumprod)) * x0_pred + \
               (torch.sqrt(alpha) * (1 - self.alphas_cumprod_prev[t]) / (1 - alpha_cumprod)) * x

        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * noise
        return mean


# =============================================================================
# Patched DDPM (pDDPM) Utilities
# =============================================================================

def create_patch_mask(image_size: int, patch_size: int, patch_pos: Tuple[int, int]) -> torch.Tensor:
    """Create a binary mask for the patch region.

    Returns mask where 1 = patch region (to be noised), 0 = context (keep clean).
    """
    mask = torch.zeros(1, image_size, image_size)
    y, x = patch_pos
    mask[:, y:y+patch_size, x:x+patch_size] = 1.0
    return mask


def get_patch_positions(image_size: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Get all patch positions for tiled reconstruction."""
    positions = []
    for y in range(0, image_size - patch_size + 1, stride):
        for x in range(0, image_size - patch_size + 1, stride):
            positions.append((y, x))
    return positions


def sample_random_patch_position(image_size: int, patch_size: int) -> Tuple[int, int]:
    """Sample a random patch position."""
    max_pos = image_size - patch_size
    y = random.randint(0, max_pos)
    x = random.randint(0, max_pos)
    return (y, x)


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    full_dataset = CTDataset(
        data_dir=args.data_dir,
        csv_path=str(csv_path),
        healthy_only=True,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed
    )

    # Split into train/val (90/10 split)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = UNet(
        in_ch=1,  # Standard training uses single channel
        out_ch=1,
        ch=args.base_channels,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(32, 16, 8),
        dropout=args.dropout
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Scheduler
    scheduler = NoiseScheduler(
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images in pbar:
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)

            # Add noise
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)

            # Predict noise
            pred = model(noisy, t)
            loss = F.mse_loss(pred, noise)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                images = images.to(device)
                batch_size = images.shape[0]

                # Sample random timesteps
                t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)

                # Add noise
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)

                # Predict noise
                pred = model(noisy, t)
                loss = F.mse_loss(pred, noise)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / "best_model.pt")
            print(f"  -> Saved new best model (val_loss: {avg_val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

            # Generate samples
            generate_samples(model, scheduler, device, output_dir / f"samples_epoch_{epoch+1}.png")

    print(f"Training complete! Best val loss: {best_val_loss:.6f}")
    return model, scheduler


def generate_samples(model, scheduler, device, output_path, num_samples=16):
    """Generate samples from the trained model."""
    model.eval()

    with torch.no_grad():
        samples = torch.randn(num_samples, 1, 256, 256, device=device)

        for t in tqdm(reversed(range(scheduler.timesteps)), desc="Sampling", leave=False):
            samples = scheduler.sample_step(model, samples, t)

        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = samples.clamp(0, 1).cpu()

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved samples to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train pDDPM for CT Anomaly Detection')

    # Data
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data',
                        help='Path to RSNA dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of training samples (None for all)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')

    # Model
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Noise schedule')

    # Misc
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data to use for validation (default: 0.1)')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == '__main__':
    main()
