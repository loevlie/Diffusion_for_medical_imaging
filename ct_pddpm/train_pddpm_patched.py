"""
Patched Diffusion Model (pDDPM) Training - Following Paper Method
==================================================================

Based on:
- "Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI"
  (Behrendt et al., MIDL 2023) - arXiv:2303.03758
- "AnoDDPM: Anomaly Detection with DDPM using Simplex Noise"
  (Wyatt et al., CVPR Workshop 2022)

Key differences from standard DDPM training:
1. Brain-centered preprocessing (crop to brain region)
2. Patch-based training: noise is added ONLY to patches, context stays clean
3. Simplex noise instead of Gaussian (captures spatial correlations better)
4. Model learns to reconstruct patches using surrounding context

Usage:
    python train_pddpm_patched.py --data_dir /media/M2SSD/gen_models_data --epochs 50
    python train_pddpm_patched.py --noise_type simplex --epochs 50
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
from scipy import ndimage

# Try to import opensimplex for simplex noise, fall back to custom implementation
try:
    from opensimplex import OpenSimplex
    HAS_OPENSIMPLEX = True
except ImportError:
    HAS_OPENSIMPLEX = False


# =============================================================================
# Simplex Noise Implementation (based on AnoDDPM)
# =============================================================================

class SimplexNoise:
    """
    Multi-scale Simplex noise generator for anomaly detection.

    Based on AnoDDPM (Wyatt et al., CVPR Workshop 2022).
    Simplex noise is smoother and captures spatial correlations better than Gaussian.

    Reference: https://github.com/Julian-Wyatt/AnoDDPM
    """

    def __init__(self, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        if HAS_OPENSIMPLEX:
            self.simplex = OpenSimplex(seed=seed)
        else:
            # Fallback: use Perlin-like noise approximation
            self.seed = seed
            np.random.seed(seed)
            self.simplex = None

    def _noise2d_opensimplex(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate 2D simplex noise using opensimplex library."""
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = self.simplex.noise2(x[i, j], y[i, j])
        return result

    def _noise2d_fallback(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fallback noise using interpolated random values (Perlin-like)."""
        # Simple Perlin-like noise approximation
        # This is less accurate but works without dependencies

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(a, b, t):
            return a + t * (b - a)

        def grad(hash_val, x, y):
            h = hash_val & 3
            if h == 0:
                return x + y
            elif h == 1:
                return -x + y
            elif h == 2:
                return x - y
            else:
                return -x - y

        # Create permutation table
        np.random.seed(self.seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.concatenate([p, p])

        result = np.zeros_like(x, dtype=np.float64)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                xi, yi = x[i, j], y[i, j]
                X = int(np.floor(xi)) & 255
                Y = int(np.floor(yi)) & 255
                xi -= np.floor(xi)
                yi -= np.floor(yi)
                u = fade(xi)
                v = fade(yi)

                A = p[X] + Y
                B = p[X + 1] + Y

                result[i, j] = lerp(
                    lerp(grad(p[A], xi, yi), grad(p[B], xi - 1, yi), u),
                    lerp(grad(p[A + 1], xi, yi - 1), grad(p[B + 1], xi - 1, yi - 1), u),
                    v
                )

        return result

    def generate_noise_2d(self, shape: Tuple[int, int], scale: float = 32.0) -> np.ndarray:
        """Generate 2D simplex noise at given scale."""
        h, w = shape
        y_coords, x_coords = np.meshgrid(
            np.arange(h) / scale,
            np.arange(w) / scale,
            indexing='ij'
        )

        if self.simplex is not None:
            noise = self._noise2d_opensimplex(x_coords, y_coords)
        else:
            noise = self._noise2d_fallback(x_coords, y_coords)

        return noise.astype(np.float32)

    def generate_octave_noise_2d(self, shape: Tuple[int, int],
                                  octaves: int = 4,
                                  persistence: float = 0.5,
                                  base_scale: float = 32.0) -> np.ndarray:
        """
        Generate multi-scale (octave) simplex noise.

        This creates noise with features at multiple scales, which is better
        for capturing anomalies of different sizes.

        Args:
            shape: (height, width)
            octaves: Number of noise layers
            persistence: Amplitude decay per octave (0.5 = halve each time)
            base_scale: Starting scale

        Returns:
            Noise array normalized to roughly [-1, 1]
        """
        noise = np.zeros(shape, dtype=np.float32)
        amplitude = 1.0
        total_amplitude = 0.0
        scale = base_scale

        for _ in range(octaves):
            noise += amplitude * self.generate_noise_2d(shape, scale)
            total_amplitude += amplitude
            amplitude *= persistence
            scale /= 2  # Double the frequency each octave

        # Normalize
        noise /= total_amplitude

        return noise


def generate_simplex_noise_batch(batch_size: int, channels: int, height: int, width: int,
                                  octaves: int = 4, device: str = 'cpu') -> torch.Tensor:
    """
    Generate a batch of simplex noise tensors.

    This is slower than Gaussian but produces more structured noise that
    better matches natural image patterns.
    """
    noise_batch = []

    for _ in range(batch_size):
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

        noise_batch.append(np.stack(noise_channels, axis=0))

    noise_tensor = torch.tensor(np.stack(noise_batch, axis=0), dtype=torch.float32)

    # CRITICAL: Normalize simplex noise to have std=1.0 like Gaussian noise
    # Raw simplex noise has std≈0.22, but DDPM expects std=1.0
    noise_std = noise_tensor.std()
    if noise_std > 0:
        noise_tensor = noise_tensor / noise_std

    return noise_tensor.to(device)


# =============================================================================
# Brain-Centered CT Preprocessing
# =============================================================================

def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """Apply CT windowing to convert HU values to displayable range."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed.astype(np.float32)


def load_and_preprocess_dicom(filepath: str, image_size: int = 256, center_on_brain: bool = True) -> Optional[np.ndarray]:
    """Load DICOM, apply brain window, and optionally center on brain region."""
    try:
        dcm = pydicom.dcmread(filepath)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Get rescale parameters for HU conversion
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        slope = float(getattr(dcm, 'RescaleSlope', 1))

        # Convert to Hounsfield Units
        hu_image = pixel_array * slope + intercept

        # Apply brain window (W:80, L:40)
        windowed = apply_window(hu_image, window_center=40, window_width=80)

        # Center on brain region (conservative cropping to avoid over-cropping)
        if center_on_brain:
            # Include skull in mask to avoid cropping too tight
            # Brain + skull roughly HU -50 to 150
            brain_mask = (hu_image > -50) & (hu_image < 150)

            if brain_mask.any():
                # Clean up mask - be generous to include surrounding tissue
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
                        rmax = min(windowed.shape[0], rmax + pad)
                        cmin = max(0, cmin - pad)
                        cmax = min(windowed.shape[1], cmax + pad)

                        crop_h = rmax - rmin
                        crop_w = cmax - cmin
                        orig_size = min(windowed.shape)

                        # Only crop if keeping at least 70% of original
                        # This prevents over-cropping on unusual slices
                        if crop_h >= orig_size * 0.7 and crop_w >= orig_size * 0.7:
                            windowed = windowed[rmin:rmax, cmin:cmax]

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
# Dataset with Patch Extraction
# =============================================================================

class CTPatchDataset(Dataset):
    """Dataset for CT scans with patch-based training.

    Instead of returning whole images, this returns:
    - The full image (for context)
    - A random patch position

    During training, noise is added only to the patch region.
    """

    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 healthy_only: bool = True,
                 image_size: int = 256,
                 patch_size: int = 64,
                 center_on_brain: bool = True,
                 num_samples: Optional[int] = None,
                 seed: int = 42):

        self.data_dir = Path(data_dir) / "stage_2_train"
        self.image_size = image_size
        self.patch_size = patch_size
        self.center_on_brain = center_on_brain

        # Parse labels
        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])

        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        if healthy_only:
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

        # Pre-calculate patch grid positions
        self.patch_positions = self._get_patch_grid_positions()
        print(f"Patch grid: {len(self.patch_positions)} positions per image")

    def _get_patch_grid_positions(self, stride: int = None) -> List[Tuple[int, int]]:
        """Get evenly spaced patch positions (grid pattern)."""
        if stride is None:
            stride = self.patch_size // 2  # 50% overlap

        positions = []
        for y in range(0, self.image_size - self.patch_size + 1, stride):
            for x in range(0, self.image_size - self.patch_size + 1, stride):
                positions.append((y, x))
        return positions

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, idx):
        uid = self.sop_uids[idx]
        filepath = self.data_dir / f"ID_{uid}.dcm"

        image = load_and_preprocess_dicom(
            str(filepath),
            self.image_size,
            center_on_brain=self.center_on_brain
        )

        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Add channel dimension [H, W] -> [1, H, W]
        image = torch.tensor(image).unsqueeze(0)

        # Sample a random patch position from the grid
        patch_pos = random.choice(self.patch_positions)

        return image, torch.tensor(patch_pos)


# =============================================================================
# Model Architecture (same UNet as original)
# =============================================================================

class SinusoidalEmbedding(nn.Module):
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
        out = torch.einsum('bnde,bnce->bncd', attn, v)
        out = out.reshape(b, c, h, w)

        return x + self.proj(out)


class UNet(nn.Module):
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
            in_c = rev_channels[i]
            skip_c = rev_channels[i + 1]
            out_c = rev_channels[i + 1]

            blocks = nn.ModuleList()
            blocks.append(ResBlock(in_c + skip_c, out_c, time_dim, dropout))
            for j in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_c, out_c, time_dim, dropout))

            self.up_blocks.append(blocks)

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

        skips = [h]
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if downsample is not None:
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

            if upsample is not None:
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

    def add_noise_to_patch(self, x, noise, t, patch_pos, patch_size):
        """Add noise ONLY to the patch region, keep context clean.

        This is the key difference from standard DDPM training!
        The model learns to reconstruct noisy patches using clean context.
        """
        batch_size = x.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Create noisy version
        noisy = x.clone()

        for i in range(batch_size):
            y, xx = patch_pos[i]
            y, xx = int(y), int(xx)

            # Get the patch region
            original_patch = x[i, :, y:y+patch_size, xx:xx+patch_size]
            noise_patch = noise[i, :, y:y+patch_size, xx:xx+patch_size]

            # Add noise only to the patch
            noisy_patch = sqrt_alpha[i] * original_patch + sqrt_one_minus[i] * noise_patch
            noisy[i, :, y:y+patch_size, xx:xx+patch_size] = noisy_patch

        return noisy


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("PATCHED pDDPM TRAINING (Paper Method)")
    print(f"{'='*60}")
    print(f"Key features:")
    print(f"  - Brain-centered preprocessing: {args.center_on_brain}")
    print(f"  - Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  - Noise type: {args.noise_type}")
    print(f"  - Noise added ONLY to patches, context stays clean")
    print(f"{'='*60}\n")

    if args.noise_type == 'simplex' and not HAS_OPENSIMPLEX:
        print("WARNING: opensimplex not installed, using Perlin-like fallback")
        print("Install with: pip install opensimplex")

    # Dataset
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    dataset = CTPatchDataset(
        data_dir=args.data_dir,
        csv_path=str(csv_path),
        healthy_only=True,
        image_size=args.image_size,
        patch_size=args.patch_size,
        center_on_brain=args.center_on_brain,
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

    # Model
    model = UNet(
        in_ch=1,
        out_ch=1,
        ch=args.base_channels,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(32, 16, 8),
        dropout=args.dropout
    ).to(device)

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
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, patch_positions in pbar:
            images = images.to(device)
            patch_positions = patch_positions.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps (around the paper's recommended t=350)
            # We sample from a range to help the model learn different noise levels
            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)

            # Generate noise (Gaussian or Simplex based on args)
            if args.noise_type == 'simplex':
                noise = generate_simplex_noise_batch(
                    batch_size, 1, args.image_size, args.image_size,
                    octaves=args.simplex_octaves, device=device
                )
            else:
                noise = torch.randn_like(images)

            # Add noise ONLY to patches (key pDDPM difference!)
            noisy = scheduler.add_noise_to_patch(
                images, noise, t, patch_positions, args.patch_size
            )

            # Predict x0 directly (model sees noisy patch + clean context)
            # This follows the paper's approach: objective='pred_x0'
            pred_x0 = model(noisy, t)

            # Loss only on the patch region (where noise was added)
            # Target is the ORIGINAL clean image (x0), not the noise
            loss = 0
            for i in range(batch_size):
                y, x = patch_positions[i]
                y, x = int(y), int(x)
                patch_pred = pred_x0[i, :, y:y+args.patch_size, x:x+args.patch_size]
                patch_target = images[i, :, y:y+args.patch_size, x:x+args.patch_size]  # Original clean patch
                loss += F.mse_loss(patch_pred, patch_target)
            loss = loss / batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'patch_size': args.patch_size,
                'center_on_brain': args.center_on_brain,
                'noise_type': args.noise_type,
                'objective': 'pred_x0',  # Paper's approach: predict x0 directly
            }, output_dir / "best_model.pt")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'patch_size': args.patch_size,
                'center_on_brain': args.center_on_brain,
                'noise_type': args.noise_type,
                'objective': 'pred_x0',  # Paper's approach: predict x0 directly
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

            # Generate visualization
            visualize_patch_reconstruction(
                model, scheduler, dataset, device, args.patch_size,
                output_dir / f"recon_epoch_{epoch+1}.png"
            )

    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    return model, scheduler


def visualize_patch_reconstruction(model, scheduler, dataset, device, patch_size, output_path, num_samples=4):
    """Visualize patch reconstruction."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            image, patch_pos = dataset[random.randint(0, len(dataset)-1)]
            image = image.unsqueeze(0).to(device)
            y, x = int(patch_pos[0]), int(patch_pos[1])

            # Add noise at t=350 (paper's recommended level)
            t = torch.tensor([350], device=device)
            noise = torch.randn_like(image)

            # Create patched noisy input
            sqrt_alpha = scheduler.sqrt_alphas_cumprod[350]
            sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[350]

            noisy = image.clone()
            original_patch = image[:, :, y:y+patch_size, x:x+patch_size]
            noise_patch = noise[:, :, y:y+patch_size, x:x+patch_size]
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * noise_patch
            noisy[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            # Predict x0 directly (paper's approach)
            x0_pred = model(noisy, t)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # Convert to displayable
            orig = (image[0, 0].cpu().numpy() + 1) / 2
            noisy_img = (noisy[0, 0].cpu().numpy() + 1) / 2
            recon = (x0_pred[0, 0].cpu().numpy() + 1) / 2
            diff = np.abs(orig - recon)

            # Plot
            axes[i, 0].imshow(orig, cmap='gray')
            axes[i, 0].set_title('Original')
            rect = plt.Rectangle((x, y), patch_size, patch_size, fill=False, color='cyan', linewidth=2)
            axes[i, 0].add_patch(rect)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(noisy_img, cmap='gray')
            axes[i, 1].set_title('Noisy Patch (t=350)')
            rect = plt.Rectangle((x, y), patch_size, patch_size, fill=False, color='red', linewidth=2)
            axes[i, 1].add_patch(rect)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(recon, cmap='gray')
            axes[i, 2].set_title('Reconstructed')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            axes[i, 3].set_title('Difference')
            axes[i, 3].axis('off')

    plt.suptitle('Patched pDDPM Training Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Patched pDDPM (Paper Method)')

    # Data
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_patched')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=256)

    # Patched pDDPM specific
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size (paper uses ~60)')
    parser.add_argument('--center_on_brain', action='store_true', default=True,
                        help='Center images on brain region')
    parser.add_argument('--no_center_on_brain', action='store_false', dest='center_on_brain')

    # Noise type (AnoDDPM uses simplex noise)
    parser.add_argument('--noise_type', type=str, default='simplex',
                        choices=['gaussian', 'simplex'],
                        help='Noise type: gaussian (standard) or simplex (AnoDDPM)')
    parser.add_argument('--simplex_octaves', type=int, default=4,
                        help='Number of octaves for simplex noise (more = finer detail)')

    # Model
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'])

    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
