"""
Patched Diffusion Model (pDDPM) - Training FROM HUGGINGFACE PRETRAINED MODEL
=============================================================================

Based on: "Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI"
(Behrendt et al., MIDL 2023) - arXiv:2303.03758

This version uses a pretrained UNet from HuggingFace Diffusers (e.g., google/ddpm-celebahq-256)
and adapts it for grayscale CT brain scans by modifying the input/output channels.

Usage:
    python train_from_pretrained.py --data_dir /media/M2SSD/gen_models_data
    python train_from_pretrained.py --hf_model google/ddpm-ema-celebahq-256 --data_dir /media/M2SSD/gen_models_data
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
# HuggingFace Model Adapter
# =============================================================================

def adapt_unet_for_grayscale(unet):
    """
    Adapt a pretrained RGB UNet (3 channels) for grayscale (1 channel).

    Strategy:
    - Input: Average the 3 input channel weights to create 1 channel
    - Output: Average the 3 output channel weights to create 1 channel
    """
    # Adapt input convolution (3 -> 1 channel)
    old_conv_in = unet.conv_in
    new_conv_in = nn.Conv2d(
        1, old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding
    )

    # Average the RGB weights to create grayscale weights
    with torch.no_grad():
        new_conv_in.weight.data = old_conv_in.weight.data.mean(dim=1, keepdim=True)
        new_conv_in.bias.data = old_conv_in.bias.data.clone()

    unet.conv_in = new_conv_in

    # Adapt output convolution (3 -> 1 channel)
    old_conv_out = unet.conv_out
    new_conv_out = nn.Conv2d(
        old_conv_out.in_channels, 1,
        kernel_size=old_conv_out.kernel_size,
        stride=old_conv_out.stride,
        padding=old_conv_out.padding
    )

    # Average the RGB weights to create grayscale weights
    with torch.no_grad():
        new_conv_out.weight.data = old_conv_out.weight.data.mean(dim=0, keepdim=True)
        new_conv_out.bias.data = old_conv_out.bias.data[:1].clone()

    unet.conv_out = new_conv_out

    # Update config
    unet.config['in_channels'] = 1
    unet.config['out_channels'] = 1

    return unet


def load_pretrained_unet(model_id: str, device: str = 'cpu'):
    """Load a pretrained UNet from HuggingFace and adapt for grayscale."""
    from diffusers import UNet2DModel

    print(f"Loading pretrained UNet from: {model_id}")
    unet = UNet2DModel.from_pretrained(model_id)

    # Check if adaptation is needed
    if unet.config.in_channels == 3:
        print("Adapting RGB model (3 channels) -> Grayscale (1 channel)")
        unet = adapt_unet_for_grayscale(unet)
    elif unet.config.in_channels == 1:
        print("Model already configured for grayscale (1 channel)")
    else:
        raise ValueError(f"Unexpected input channels: {unet.config.in_channels}")

    unet = unet.to(device)

    # Print model info
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Model sample size: {unet.config.sample_size}")

    return unet


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

        # HuggingFace UNet returns a dict with 'sample' key
        output = model(x, t_tensor)
        if hasattr(output, 'sample'):
            pred_noise = output.sample
        else:
            pred_noise = output

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

    # Split into train/val
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

    # Load pretrained model from HuggingFace
    model = load_pretrained_unet(args.hf_model, device)

    # Scheduler
    scheduler = NoiseScheduler(
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device
    )

    # Optimizer (lower LR for fine-tuning)
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

            # Predict noise (HuggingFace UNet returns dict)
            output = model(noisy, t)
            if hasattr(output, 'sample'):
                pred = output.sample
            else:
                pred = output

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
                output = model(noisy, t)
                if hasattr(output, 'sample'):
                    pred = output.sample
                else:
                    pred = output

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
                'hf_model': args.hf_model,
                'config': dict(model.config),
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
                'hf_model': args.hf_model,
                'config': dict(model.config),
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
    parser = argparse.ArgumentParser(description='Train pDDPM from HuggingFace Pretrained Model')

    # HuggingFace model
    parser.add_argument('--hf_model', type=str, default='google/ddpm-ema-celebahq-256',
                        help='HuggingFace model ID (e.g., google/ddpm-celebahq-256, google/ddpm-ema-celebahq-256)')

    # Data
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data',
                        help='Path to RSNA dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_pretrained',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of training samples (None for all)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='linear',
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
