"""
Patched pDDPM Evaluation for CT Hemorrhage Detection
====================================================

Usage:
    # Evaluate from-scratch model
    python evaluate.py --checkpoint checkpoints_scratch/best_model.pt

    # Evaluate pretrained model
    python evaluate.py --checkpoint checkpoints_pretrained/best_model.pt
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

from train import UNet, NoiseScheduler, load_and_preprocess_dicom


class Evaluator:
    def __init__(self, model, scheduler, device, is_hf=False, num_channels=1):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.is_hf = is_hf
        self.num_channels = num_channels  # Model's expected input channels

    @torch.no_grad()
    def one_step_denoise(self, noisy_image, t):
        """One-step denoising: predict noise, compute x0 directly."""
        t_tensor = torch.full((noisy_image.shape[0],), t, device=self.device, dtype=torch.long)
        output = self.model(noisy_image, t_tensor)
        pred_noise = output.sample if self.is_hf and hasattr(output, 'sample') else output

        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (noisy_image - sqrt_one_minus * pred_noise) / sqrt_alpha
        return torch.clamp(x0_pred, -1, 1)

    @torch.no_grad()
    def patched_anomaly_detection(self, image_tensor, patch_size=64, stride=32, noise_level=350):
        """Compute anomaly map using patched reconstruction with one-step denoising."""
        B, C, H, W = image_tensor.shape

        # Replicate grayscale to 3 channels if model expects 3 channels
        if self.num_channels == 3 and C == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
            C = 3

        anomaly_map = torch.zeros((H, W), device=self.device)
        count_map = torch.zeros((H, W), device=self.device)

        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[noise_level]
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patched_input = image_tensor.clone()
                patch_noise = torch.randn(B, C, patch_size, patch_size, device=self.device)
                original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].clone()
                noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
                patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

                reconstructed = self.one_step_denoise(patched_input, noise_level)
                recon_patch = reconstructed[:, :, y:y+patch_size, x:x+patch_size]

                # Average across channels for error computation
                patch_error = torch.abs(original_patch - recon_patch).mean(dim=1).squeeze()

                anomaly_map[y:y+patch_size, x:x+patch_size] += patch_error
                count_map[y:y+patch_size, x:x+patch_size] += 1

        return anomaly_map / torch.clamp(count_map, min=1)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if HuggingFace model by looking at state dict keys
    state_dict = checkpoint['model_state_dict']
    has_unet_prefix = any(k.startswith('unet.') for k in state_dict.keys())
    is_hf = checkpoint.get('is_hf', 'hf_model' in checkpoint or has_unet_prefix)

    if is_hf:
        from diffusers import UNet2DModel

        hf_model_id = checkpoint.get('hf_model', 'google/ddpm-ema-celebahq-256')

        # Handle state dict with 'unet.' prefix
        if has_unet_prefix:
            state_dict = {k.replace('unet.', ''): v for k, v in state_dict.items() if k.startswith('unet.')}

        # Check if saved weights are grayscale (1ch) or RGB (3ch)
        conv_in_shape = state_dict['conv_in.weight'].shape
        in_channels = conv_in_shape[1]  # Shape is [out_ch, in_ch, H, W]

        # Create model with correct channel config
        base_model = UNet2DModel.from_pretrained(hf_model_id)
        config = dict(base_model.config)
        config['in_channels'] = in_channels
        config['out_channels'] = in_channels
        model = UNet2DModel(**config).to(device)

        model.load_state_dict(state_dict)
        num_channels = in_channels
        print(f"Loaded HuggingFace model (base: {hf_model_id}, channels: {in_channels})")
    else:
        model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
        model.load_state_dict(state_dict)
        num_channels = 1
        print("Loaded custom UNet")

    model.eval()
    print(f"Checkpoint epoch: {checkpoint['epoch']}, val_loss: {checkpoint.get('val_loss', 'N/A')}")
    return model, is_hf, num_channels


def main():
    parser = argparse.ArgumentParser(description='Evaluate pDDPM for CT Hemorrhage Detection')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--num_healthy', type=int, default=100)
    parser.add_argument('--num_hemorrhage', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--noise_level', type=int, default=350)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, is_hf, num_channels = load_model(args.checkpoint, device)
    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)
    evaluator = Evaluator(model, scheduler, device, is_hf, num_channels)

    # Load data
    csv_path = Path(args.data_dir) / "stage_2_train.csv"
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage = pivot[pivot['any'] == 1].index.tolist()

    random.seed(args.seed)
    test_healthy = random.sample(healthy, min(args.num_healthy, len(healthy)))
    test_hemorrhage = random.sample(hemorrhage, min(args.num_hemorrhage, len(hemorrhage)))

    data_dir = Path(args.data_dir) / "stage_2_train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing: {len(test_healthy)} healthy + {len(test_hemorrhage)} hemorrhage")

    # Evaluate
    results = []
    all_samples = [(u, 0) for u in test_healthy] + [(u, 1) for u in test_hemorrhage]
    random.shuffle(all_samples)

    for uid, label in tqdm(all_samples, desc="Evaluating"):
        filepath = data_dir / f"ID_{uid}.dcm"
        if not filepath.exists():
            continue

        image = load_and_preprocess_dicom(str(filepath), 256)
        if image is None:
            continue

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
        anomaly_map = evaluator.patched_anomaly_detection(
            image_tensor, args.patch_size, args.stride, args.noise_level
        )

        results.append({
            'uid': uid,
            'label': label,
            'anomaly_score': anomaly_map.mean().item(),
        })

    # Calculate metrics
    labels = np.array([r['label'] for r in results])
    scores = np.array([r['anomaly_score'] for r in results])

    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Healthy mean score:    {np.mean(scores[labels==0]):.4f}")
    print(f"Hemorrhage mean score: {np.mean(scores[labels==1]):.4f}")
    print(f"{'='*50}")

    # Save results
    pd.DataFrame(results).to_csv(output_dir / 'results.csv', index=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - pDDPM Hemorrhage Detection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
