"""
Patched Diffusion Model (pDDPM) for CT Anomaly Detection - Pretrained Version
==============================================================================

Fine-tunes a pretrained DDPM model (google/ddpm-celebahq-256) on CT scans.

Strategy:
- Load pretrained 256x256 RGB DDPM weights
- Adapt first conv layer: 3 channels -> 1 channel (grayscale CT)
- Adapt final conv layer: 3 channels -> 1 channel
- Fine-tune on healthy CT scans with lower learning rate

This leverages learned image priors from natural images to speed up training.

Usage:
    python train_pddpm_pretrained.py --num_samples 10000 --epochs 50
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import random

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# For loading pretrained model
from diffusers import UNet2DModel, DDPMScheduler


# =============================================================================
# CT Preprocessing (same as train_pddpm.py)
# =============================================================================

def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed.astype(np.float32)


def load_and_preprocess_dicom(filepath: str, image_size: int = 256) -> Optional[np.ndarray]:
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
    """CT Dataset - healthy scans only for training."""

    def __init__(self, data_dir: str, csv_path: str, healthy_only: bool = True,
                 image_size: int = 256, num_samples: Optional[int] = None,
                 use_rgb: bool = True, seed: int = 42):
        """
        Args:
            use_rgb: If True, convert grayscale to 3-channel RGB for pretrained model
        """
        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size
        self.use_rgb = use_rgb

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

        valid_uids = []
        for uid in tqdm(self.sop_uids, desc="Checking files"):
            if (self.data_dir / f"ID_{uid}.dcm").exists():
                valid_uids.append(uid)
        self.sop_uids = valid_uids

        print(f"Dataset: {len(self.sop_uids)} scans (use_rgb={use_rgb})")

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        uid = self.sop_uids[idx]
        filepath = self.data_dir / f"ID_{uid}.dcm"

        image = load_and_preprocess_dicom(str(filepath), self.image_size)
        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.use_rgb:
            # Convert grayscale to 3-channel for pretrained model
            image = np.stack([image, image, image], axis=0)  # [3, H, W]
        else:
            image = image[np.newaxis, ...]  # [1, H, W]

        return torch.tensor(image)


# =============================================================================
# Model Adaptation
# =============================================================================

def load_pretrained_model(pretrained_name: str = "google/ddpm-celebahq-256",
                          adapt_channels: bool = True,
                          device: str = "cuda"):
    """
    Load pretrained DDPM and optionally adapt for grayscale.

    Args:
        pretrained_name: HuggingFace model name
        adapt_channels: If True, modify first/last conv for 1-channel input/output
        device: Device to load model on
    """
    print(f"Loading pretrained model: {pretrained_name}")
    model = UNet2DModel.from_pretrained(pretrained_name)

    if adapt_channels:
        print("Adapting model for grayscale (1 channel)...")

        # Adapt input conv: 3 -> 1 channel
        # Average the RGB weights to create grayscale weights
        old_conv_in = model.conv_in
        new_conv_in = nn.Conv2d(
            1, old_conv_in.out_channels,
            kernel_size=old_conv_in.kernel_size,
            stride=old_conv_in.stride,
            padding=old_conv_in.padding
        )
        # Initialize with averaged weights
        with torch.no_grad():
            new_conv_in.weight.data = old_conv_in.weight.data.mean(dim=1, keepdim=True)
            new_conv_in.bias.data = old_conv_in.bias.data.clone()
        model.conv_in = new_conv_in

        # Adapt output conv: 3 -> 1 channel
        old_conv_out = model.conv_out
        new_conv_out = nn.Conv2d(
            old_conv_out.in_channels, 1,
            kernel_size=old_conv_out.kernel_size,
            stride=old_conv_out.stride,
            padding=old_conv_out.padding
        )
        # Initialize with averaged weights
        with torch.no_grad():
            new_conv_out.weight.data = old_conv_out.weight.data.mean(dim=0, keepdim=True)
            new_conv_out.bias.data = old_conv_out.bias.data[:1].clone()
        model.conv_out = new_conv_out

        # Update config
        model.config.in_channels = 1
        model.config.out_channels = 1

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    return model


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
        use_rgb=not args.adapt_channels,  # Use grayscale if adapting
        seed=args.seed
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Load pretrained model
    model = load_pretrained_model(
        pretrained_name=args.pretrained_model,
        adapt_channels=args.adapt_channels,
        device=device
    )

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.timesteps,
        beta_schedule="squaredcos_cap_v2"
    )

    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        'pretrained_model': args.pretrained_model,
        'adapt_channels': args.adapt_channels,
        'image_size': args.image_size,
        'timesteps': args.timesteps,
    }

    # Training loop
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
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

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
                'config': config,
            }, output_dir / "best_model.pt")

        # Periodic saves
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

            generate_samples(model, scheduler, device, args, output_dir / f"samples_epoch_{epoch+1}.png")

    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")


def generate_samples(model, scheduler, device, args, output_path, num_samples=16):
    """Generate samples from trained model."""
    model.eval()
    scheduler.set_timesteps(args.timesteps)

    in_channels = 1 if args.adapt_channels else 3

    with torch.no_grad():
        samples = torch.randn(num_samples, in_channels, args.image_size, args.image_size, device=device)

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
            if in_channels == 1:
                ax.imshow(samples[i, 0], cmap='gray')
            else:
                # Convert to grayscale for display
                ax.imshow(samples[i].mean(dim=0), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved samples to {output_path}")


# =============================================================================
# Inference (for compatibility with evaluate_fast.py)
# =============================================================================

def load_model_for_inference(checkpoint_path: str, device: str = "cuda"):
    """Load trained model for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Load base pretrained model
    model = load_pretrained_model(
        pretrained_name=config.get('pretrained_model', 'google/ddpm-celebahq-256'),
        adapt_channels=config.get('adapt_channels', True),
        device=device
    )

    # Load fine-tuned weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='pDDPM Pretrained Fine-tuning')

    # Data
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_pretrained')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=256)

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, default='google/ddpm-celebahq-256',
                        choices=['google/ddpm-celebahq-256', 'google/ddpm-church-256',
                                 'google/ddpm-bedroom-256', 'google/ddpm-cifar10-32'],
                        help='Pretrained model to fine-tune')
    parser.add_argument('--adapt_channels', action='store_true', default=True,
                        help='Adapt 3-channel model to 1-channel grayscale')
    parser.add_argument('--no_adapt_channels', action='store_false', dest='adapt_channels',
                        help='Keep 3-channel input (replicate grayscale to RGB)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5,  # Lower LR for fine-tuning
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
