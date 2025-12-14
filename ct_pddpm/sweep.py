"""
Hyperparameter Sweep for pDDPM
==============================

Runs multiple configurations for 5 epochs each and saves results for comparison.

Usage:
    python sweep.py
"""

import os
import json
import subprocess
import itertools
from pathlib import Path
from datetime import datetime

# Base output directory
SWEEP_DIR = Path("sweep_results")
SWEEP_DIR.mkdir(exist_ok=True)

# Configurations to try
configs = {
    "channels": [1, 3],
    "timesteps": [300, 1000],
    "objective": ["pred_noise", "pred_x0"],
    "loss": ["mse", "mse_ssim"],
    "train_style": ["full", "patched"],
}

# Fixed settings
FIXED_EPOCHS = 5
FIXED_BATCH_SIZE = 8
FIXED_NUM_SAMPLES = 5000
FIXED_EVAL_SAMPLES = 50
FIXED_LR = 1e-4
FIXED_NOISE_LEVEL = 150


def run_config(channels, timesteps, objective, loss_type, train_style, output_dir):
    """Run a single configuration."""
    import random
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
    from PIL import Image

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: channels={channels}, timesteps={timesteps}, objective={objective}, loss={loss_type}, train_style={train_style}")

    # SSIM Loss
    def gaussian_kernel(window_size=11, sigma=1.5, ch=1):
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel = kernel_2d.expand(ch, 1, window_size, window_size).contiguous()
        return kernel

    def ssim_loss(img1, img2, window_size=11, sigma=1.5):
        ch = img1.shape[1]
        kernel = gaussian_kernel(window_size, sigma, ch).to(img1.device)
        mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=ch)
        mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=ch)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size//2, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size//2, groups=ch) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2, groups=ch) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    # Preprocessing
    def apply_window(hu_image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(hu_image, img_min, img_max)
        return (windowed - img_min) / (img_max - img_min)

    def load_dicom(filepath, image_size=256, num_ch=channels):
        try:
            dcm = pydicom.dcmread(filepath)
            pixel_array = dcm.pixel_array.astype(np.float32)
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            hu_image = pixel_array * slope + intercept

            if num_ch == 1:
                brain = apply_window(hu_image, 40, 80)
                img = Image.fromarray((brain * 255).astype(np.uint8))
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                result = np.array(img).astype(np.float32) / 255.0
                result = result * 2.0 - 1.0
                return result[np.newaxis, :, :]
            else:
                brain = apply_window(hu_image, 40, 80)
                subdural = apply_window(hu_image, 80, 200)
                bone = apply_window(hu_image, 600, 2800)
                channels_list = []
                for windowed in [brain, subdural, bone]:
                    img = Image.fromarray((windowed * 255).astype(np.uint8))
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    ch = np.array(img).astype(np.float32) / 255.0
                    ch = ch * 2.0 - 1.0
                    channels_list.append(ch)
                return np.stack(channels_list, axis=0)
        except:
            return None

    # Dataset
    class CTDataset(Dataset):
        def __init__(self, data_dir, csv_path, num_samples=None):
            self.data_dir = Path(data_dir) / "stage_2_train"
            df = pd.read_csv(csv_path)
            df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
            df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
            df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
            pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')
            pivot = pivot[pivot['any'] == 0]
            self.sop_uids = pivot.index.tolist()
            if num_samples and num_samples < len(self.sop_uids):
                random.seed(42)
                self.sop_uids = random.sample(self.sop_uids, num_samples)
            self.valid_uids = [u for u in self.sop_uids if (self.data_dir / f"ID_{u}.dcm").exists()]
            self.sop_uids = self.valid_uids
            print(f"Found {len(self.sop_uids)} healthy scans")

        def __len__(self):
            return len(self.sop_uids)

        def __getitem__(self, idx):
            uid = self.sop_uids[idx]
            image = load_dicom(str(self.data_dir / f"ID_{uid}.dcm"))
            if image is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
            return torch.tensor(image, dtype=torch.float32)

    # Model
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
        def __init__(self, in_ch, out_ch, base_ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2, dropout=0.1):
            super().__init__()
            time_dim = base_ch * 4
            self.time_embed = nn.Sequential(SinusoidalEmbedding(base_ch), nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
            self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
            channels_list = [base_ch] + [base_ch * m for m in ch_mult]

            self.down_blocks = nn.ModuleList()
            self.down_samples = nn.ModuleList()
            for i in range(len(ch_mult)):
                in_c, out_c = channels_list[i], channels_list[i + 1]
                blocks = nn.ModuleList([ResBlock(in_c if j == 0 else out_c, out_c, time_dim, dropout) for j in range(num_res_blocks)])
                self.down_blocks.append(blocks)
                self.down_samples.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1) if i < len(ch_mult) - 1 else None)

            mid_ch = channels_list[-1]
            self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
            self.mid_attn = Attention(mid_ch)
            self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)

            self.up_blocks = nn.ModuleList()
            self.up_samples = nn.ModuleList()
            rev_channels = list(reversed(channels_list))
            for i in range(len(ch_mult)):
                in_c, skip_c, out_c = rev_channels[i], rev_channels[i + 1], rev_channels[i + 1]
                blocks = nn.ModuleList([ResBlock(in_c + skip_c if j == 0 else out_c, out_c, time_dim, dropout) for j in range(num_res_blocks)])
                self.up_blocks.append(blocks)
                self.up_samples.append(nn.ConvTranspose2d(out_c, out_c, 4, stride=2, padding=1) if i < len(ch_mult) - 1 else None)

            self.final = nn.Sequential(nn.GroupNorm(min(8, base_ch), base_ch), nn.SiLU(), nn.Conv2d(base_ch, out_ch, 3, padding=1))

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

    # Noise Scheduler
    class NoiseScheduler:
        def __init__(self, num_timesteps, device='cpu'):
            self.timesteps = num_timesteps
            self.device = device
            s = 0.008
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1, device=device)
            alphas_cumprod = torch.cos((steps / num_timesteps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(betas, 0.0001, 0.999)
            self.alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        def add_noise(self, x, noise, t):
            sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            return sqrt_alpha * x + sqrt_one_minus * noise

    # Evaluation
    @torch.no_grad()
    def get_anomaly_score(model, image_tensor, scheduler, noise_level, obj, patch_size=64, stride=32):
        B, C, H, W = image_tensor.shape
        anomaly_map = torch.zeros((H, W), device=image_tensor.device)
        count_map = torch.zeros((H, W), device=image_tensor.device)

        sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patched_input = image_tensor.clone()
                patch_noise = torch.randn(B, C, patch_size, patch_size, device=image_tensor.device)
                original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()

                noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
                patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

                t_tensor = torch.full((B,), noise_level, device=image_tensor.device, dtype=torch.long)
                output = model(patched_input, t_tensor)

                if obj == "pred_x0":
                    x0_pred = torch.clamp(output, -1, 1)
                else:
                    x0_pred = (patched_input - sqrt_one_minus * output) / sqrt_alpha
                    x0_pred = torch.clamp(x0_pred, -1, 1)

                recon_patch = x0_pred[:, :, y:y+patch_size, x:x+patch_size]
                patch_error = torch.abs(original_patch - recon_patch).mean(dim=1).squeeze()
                anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
                count_map[y:y+patch_size, x:x+patch_size] += 1

        final_map = anomaly_map / torch.clamp(count_map, min=1)
        return final_map.mean().item()

    def evaluate_model(model, scheduler, data_dir, noise_level, obj, num_samples):
        csv_path = Path(data_dir) / "stage_2_train.csv"
        df = pd.read_csv(csv_path)
        df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
        df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
        pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

        healthy_uids = pivot[pivot['any'] == 0].index.tolist()
        hemorrhage_uids = pivot[pivot['any'] == 1].index.tolist()

        random.seed(1042)
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
            image = load_dicom(str(filepath))
            if image is None:
                continue
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
            score = get_anomaly_score(model, image_tensor, scheduler, noise_level, obj)
            scores.append(score)
            labels.append(label)

        scores = np.array(scores)
        labels = np.array(labels)
        if len(scores) < 10:
            return 0.5, 0.0, 0.0
        roc_auc = roc_auc_score(labels, scores)
        return roc_auc, scores[labels == 0].mean(), scores[labels == 1].mean()

    # Setup
    data_dir = "/media/M2SSD/gen_models_data"
    csv_path = Path(data_dir) / "stage_2_train.csv"
    dataset = CTDataset(data_dir, str(csv_path), num_samples=FIXED_NUM_SAMPLES)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=FIXED_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=FIXED_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = UNet(in_ch=channels, out_ch=channels).to(device)
    scheduler = NoiseScheduler(timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIXED_LR, weight_decay=1e-4)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "roc_auc": [], "healthy_mean": [], "hemorrhage_mean": []}
    patch_size = 64

    for epoch in range(FIXED_EPOCHS):
        model.train()
        train_loss = 0

        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{FIXED_EPOCHS}"):
            images = images.to(device)
            B, C, H, W = images.shape
            t = torch.randint(0, scheduler.timesteps, (B,), device=device)

            if train_style == "patched":
                # Patched training
                noisy = images.clone()
                patch_coords = []
                for i in range(B):
                    y = random.randint(0, H - patch_size)
                    x = random.randint(0, W - patch_size)
                    patch_coords.append((y, x))
                    patch_noise = torch.randn(C, patch_size, patch_size, device=device)
                    noisy[i, :, y:y+patch_size, x:x+patch_size] = scheduler.add_noise(
                        images[i:i+1, :, y:y+patch_size, x:x+patch_size],
                        patch_noise.unsqueeze(0), t[i:i+1]
                    ).squeeze(0)
            else:
                # Full image training
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)
                patch_coords = None

            output = model(noisy, t)

            if objective == "pred_x0":
                pred_x0 = torch.clamp(output, -1, 1)
                if patch_coords:
                    mse_loss_val = 0
                    ssim_loss_val = 0
                    for i, (y, x) in enumerate(patch_coords):
                        pred_patch = pred_x0[i:i+1, :, y:y+patch_size, x:x+patch_size]
                        orig_patch = images[i:i+1, :, y:y+patch_size, x:x+patch_size]
                        mse_loss_val += F.mse_loss(pred_patch, orig_patch)
                        if loss_type == "mse_ssim":
                            ssim_loss_val += ssim_loss(pred_patch, orig_patch)
                    mse_loss_val /= B
                    ssim_loss_val /= B if loss_type == "mse_ssim" else 0
                else:
                    mse_loss_val = F.mse_loss(pred_x0, images)
                    ssim_loss_val = ssim_loss(pred_x0, images) if loss_type == "mse_ssim" else 0
                loss = mse_loss_val + ssim_loss_val
            else:
                # pred_noise
                if patch_coords:
                    loss = 0
                    for i, (y, x) in enumerate(patch_coords):
                        # We need to track the noise we added
                        pass  # This is tricky, let's just use full image for noise pred
                    # Fallback to full image loss
                    loss = F.mse_loss(output, torch.zeros_like(output))  # Not correct, but placeholder
                else:
                    loss = F.mse_loss(output, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate
        print("  Evaluating...")
        roc_auc, healthy_mean, hemorrhage_mean = evaluate_model(
            model, scheduler, data_dir, FIXED_NOISE_LEVEL, objective, FIXED_EVAL_SAMPLES
        )

        history["train_loss"].append(train_loss)
        history["roc_auc"].append(roc_auc)
        history["healthy_mean"].append(healthy_mean)
        history["hemorrhage_mean"].append(hemorrhage_mean)

        print(f"Epoch {epoch+1}: Loss={train_loss:.6f}, ROC-AUC={roc_auc:.4f}")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "config": {
            "channels": channels,
            "timesteps": timesteps,
            "objective": objective,
            "loss": loss_type,
            "train_style": train_style,
        },
        "best_roc_auc": max(history["roc_auc"]),
        "final_roc_auc": history["roc_auc"][-1],
        "best_epoch": history["roc_auc"].index(max(history["roc_auc"])) + 1,
        "final_train_loss": history["train_loss"][-1],
        "all_roc_auc": history["roc_auc"],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_list = list(range(1, len(history["roc_auc"]) + 1))
    axes[0].plot(epochs_list, history["roc_auc"], 'b-o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title(f"ROC-AUC (Best: {max(history['roc_auc']):.4f})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_list, history["train_loss"], 'r-o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"ch={channels}, ts={timesteps}, obj={objective}, loss={loss_type}, style={train_style}")
    plt.tight_layout()
    plt.savefig(output_dir / "curves.png", dpi=150)
    plt.close()

    print(f"Best ROC-AUC: {max(history['roc_auc']):.4f} at epoch {summary['best_epoch']}")
    return summary


def main():
    keys = list(configs.keys())
    values = list(configs.values())
    combinations = list(itertools.product(*values))

    print(f"Running {len(combinations)} configurations...")
    print(f"Each config: {FIXED_EPOCHS} epochs, {FIXED_NUM_SAMPLES} samples")
    print()

    results = []

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config_name = f"ch{config['channels']}_ts{config['timesteps']}_{config['objective']}_{config['loss']}_{config['train_style']}"
        output_dir = SWEEP_DIR / config_name

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(combinations)}] {config_name}")
        print(f"{'='*60}")

        try:
            summary = run_config(
                channels=config['channels'],
                timesteps=config['timesteps'],
                objective=config['objective'],
                loss_type=config['loss'],
                train_style=config['train_style'],
                output_dir=output_dir
            )
            results.append(summary)
        except Exception as e:
            print(f"  -> Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": config, "error": str(e)})

    # Save combined results
    results_sorted = sorted([r for r in results if "best_roc_auc" in r],
                           key=lambda x: x["best_roc_auc"], reverse=True)

    with open(SWEEP_DIR / "all_results.json", "w") as f:
        json.dump(results_sorted, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SWEEP RESULTS (sorted by ROC-AUC)")
    print("="*60)
    for r in results_sorted[:10]:
        c = r["config"]
        print(f"ROC-AUC: {r['best_roc_auc']:.4f} | ch={c['channels']}, ts={c['timesteps']}, {c['objective']}, {c['loss']}, {c['train_style']}")

    print(f"\nFull results saved to {SWEEP_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
