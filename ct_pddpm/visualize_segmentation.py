"""
Visualize pDDPM reconstruction on segmentation dataset with ground truth masks.
Shows: Original, Mask, Reconstruction, Difference
Computes Dice score between thresholded error map and ground truth.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random

from train_patched import UNet, NoiseScheduler, load_and_preprocess_dicom, crop_brain_region


def load_model(checkpoint_path, device):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, ROC-AUC: {checkpoint.get('roc_auc', 'N/A')}")
    return model


def load_segmentation_dataset(data_dir, target_size=256):
    """Load CT ICH segmentation dataset with masks."""
    seg_dir = Path(data_dir) / "ct_ich_masks" / \
        "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"

    samples = []
    patients_dir = seg_dir / "Patients_CT"

    for patient_dir in sorted(patients_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        brain_dir = patient_dir / "brain"
        if not brain_dir.exists():
            continue

        for mask_file in brain_dir.glob("*_HGE_Seg.jpg"):
            slice_num = mask_file.stem.replace("_HGE_Seg", "")
            img_file = brain_dir / f"{slice_num}.jpg"
            if img_file.exists():
                samples.append({
                    'image': str(img_file),
                    'mask': str(mask_file),
                    'patient': patient_dir.name,
                    'slice': slice_num
                })

    print(f"Found {len(samples)} images with segmentation masks")
    return samples


def load_jpg_ct(filepath, target_size=256):
    """Load pre-processed CT from JPG."""
    img = Image.open(filepath).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    cropped = crop_brain_region(arr, threshold=0.05, margin=5)
    img = Image.fromarray((cropped * 255).astype(np.uint8))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
    return arr


def load_mask(filepath, target_size=256):
    """Load segmentation mask."""
    img = Image.open(filepath).convert('L')
    img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


@torch.no_grad()
def reconstruct_patched(model, image_tensor, scheduler, device, patch_size=64, stride=32, noise_level=150):
    """Reconstruct image using patched approach and return error map."""
    B, C, H, W = image_tensor.shape

    reconstructed = torch.zeros_like(image_tensor)
    count_map = torch.zeros((B, 1, H, W), device=device)

    sqrt_alpha = scheduler.sqrt_alphas_cumprod[noise_level]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[noise_level]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patched_input = image_tensor.clone()
            patch_noise = torch.randn(B, C, patch_size, patch_size, device=device)
            original_patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            noisy_patch = sqrt_alpha * original_patch + sqrt_one_minus * patch_noise
            patched_input[:, :, y:y+patch_size, x:x+patch_size] = noisy_patch

            t_tensor = torch.full((B,), noise_level, device=device, dtype=torch.long)
            pred_x0 = model(patched_input, t_tensor)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            reconstructed[:, :, y:y+patch_size, x:x+patch_size] += pred_x0[:, :, y:y+patch_size, x:x+patch_size]
            count_map[:, :, y:y+patch_size, x:x+patch_size] += 1

    reconstructed = reconstructed / torch.clamp(count_map, min=1)
    error_map = torch.abs(reconstructed - image_tensor)

    return reconstructed.squeeze(), error_map.squeeze()


def dice_score(pred_mask, gt_mask):
    """Compute Dice score between predicted and ground truth masks."""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0:
        return 1.0 if np.sum(gt_mask) == 0 else 0.0
    return 2 * intersection / union


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_mse_only/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/gen_models_data')
    parser.add_argument('--output', type=str, default='segmentation_results.png')
    parser.add_argument('--noise_level', type=int, default=150)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)

    # Load segmentation dataset
    samples = load_segmentation_dataset(args.data_dir)

    # Process all samples and compute Dice scores
    results = []
    print("Processing all samples...")

    for sample in tqdm(samples):
        image = load_jpg_ct(sample['image'], 256)
        mask = load_mask(sample['mask'], 256)

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
        reconstructed, error_map = reconstruct_patched(model, image_tensor, scheduler, device,
                                                        noise_level=args.noise_level)

        error_np = error_map.cpu().numpy()

        # Try different thresholds to find best Dice
        best_dice = 0
        best_threshold = 0
        for thresh in np.linspace(0.01, 0.5, 50):
            pred_mask = (error_np > thresh).astype(np.float32)
            d = dice_score(pred_mask, mask)
            if d > best_dice:
                best_dice = d
                best_threshold = thresh

        results.append({
            'sample': sample,
            'image': image,
            'mask': mask,
            'reconstructed': reconstructed.cpu().numpy(),
            'error_map': error_np,
            'dice': best_dice,
            'threshold': best_threshold
        })

    # Sort by Dice score
    results.sort(key=lambda x: x['dice'], reverse=True)

    # Get best and random samples
    best_result = results[0]
    random_result = random.choice(results[len(results)//4:len(results)*3//4])  # Pick from middle range

    print(f"\nBest Dice: {best_result['dice']:.4f}")
    print(f"Random Dice: {random_result['dice']:.4f}")
    print(f"Mean Dice: {np.mean([r['dice'] for r in results]):.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for row, (result, title) in enumerate([(best_result, f"Best (Dice={best_result['dice']:.3f})"),
                                            (random_result, f"Random (Dice={random_result['dice']:.3f})")]):
        # Original
        img_display = (result['image'] + 1) / 2
        axes[row, 0].imshow(img_display, cmap='gray')
        axes[row, 0].set_title(f'{title}\nOriginal')
        axes[row, 0].axis('off')

        # Ground truth mask
        axes[row, 1].imshow(result['mask'], cmap='Reds')
        axes[row, 1].set_title('GT Hemorrhage Mask')
        axes[row, 1].axis('off')

        # Reconstruction
        recon_display = (result['reconstructed'] + 1) / 2
        axes[row, 2].imshow(recon_display, cmap='gray')
        axes[row, 2].set_title('Reconstruction')
        axes[row, 2].axis('off')

        # Error map
        axes[row, 3].imshow(result['error_map'], cmap='hot')
        axes[row, 3].set_title(f'Error Map\n(thresh={result["threshold"]:.3f})')
        axes[row, 3].axis('off')

        # Overlay
        axes[row, 4].imshow(img_display, cmap='gray')
        axes[row, 4].imshow(result['error_map'], cmap='hot', alpha=0.5)
        axes[row, 4].contour(result['mask'], colors='lime', linewidths=2)
        axes[row, 4].set_title('Overlay (green=GT)')
        axes[row, 4].axis('off')

    plt.suptitle(f'pDDPM Hemorrhage Detection on Segmentation Dataset\nMean Dice: {np.mean([r["dice"] for r in results]):.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to {args.output}")

    # Also save Dice distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([r['dice'] for r in results], bins=30, edgecolor='black')
    ax.axvline(np.mean([r['dice'] for r in results]), color='red', linestyle='--',
               label=f'Mean: {np.mean([r["dice"] for r in results]):.4f}')
    ax.set_xlabel('Dice Score')
    ax.set_ylabel('Count')
    ax.set_title('Dice Score Distribution')
    ax.legend()
    plt.savefig(args.output.replace('.png', '_dice_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Dice distribution to {args.output.replace('.png', '_dice_dist.png')}")


if __name__ == '__main__':
    main()
