"""
Create presentation figures for CT hemorrhage detection comparison.
Generates:
1. Training curves comparing scratch vs pretrained models
2. Hemorrhage visualization with original, reconstruction, difference map, and mask outline
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import random
from tqdm import tqdm

# Import from train
from train_patched import UNet, NoiseScheduler, crop_brain_region

# ================================================================================
# Configuration
# ================================================================================

SCRATCH_DIR = Path("checkpoints_patched")
PRETRAINED_DIR = Path("checkpoints_pretrained")
OUTPUT_DIR = Path("presentation_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("/media/M2SSD/gen_models_data")
MASK_DIR = DATA_DIR / "ct_ich_masks" / \
    "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"

# ================================================================================
# 1. Training Curves Comparison Plot
# ================================================================================

def create_training_curves():
    """Create side-by-side comparison plots for validation loss and ROC-AUC."""

    # Load training histories
    with open(SCRATCH_DIR / "training_history.json") as f:
        scratch_hist = json.load(f)

    with open(PRETRAINED_DIR / "training_history.json") as f:
        pretrained_hist = json.load(f)

    # Limit both to 10 epochs for fair comparison
    max_epochs = 10
    for key in scratch_hist:
        scratch_hist[key] = scratch_hist[key][:max_epochs]
    for key in pretrained_hist:
        pretrained_hist[key] = pretrained_hist[key][:max_epochs]

    # Get epochs (both now have 10)
    scratch_epochs = list(range(1, len(scratch_hist['roc_auc']) + 1))
    pretrained_epochs = list(range(1, len(pretrained_hist['roc_auc']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Validation Loss
    ax1 = axes[0]
    ax1.plot(scratch_epochs, scratch_hist['val_loss'], 'b-o', linewidth=2, markersize=6,
             label=f"Scratch UNet (~30M params)", alpha=0.8)
    ax1.plot(pretrained_epochs, pretrained_hist['val_loss'], 'r-s', linewidth=2, markersize=6,
             label=f"Pretrained (~114M params)", alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss (MAE)', fontsize=12)
    ax1.set_title('Validation Loss Over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.5, max(len(scratch_epochs), len(pretrained_epochs)) + 0.5)

    # Plot 2: ROC-AUC
    ax2 = axes[1]
    ax2.plot(scratch_epochs, scratch_hist['roc_auc'], 'b-o', linewidth=2, markersize=6,
             label=f"Scratch UNet (Best: {max(scratch_hist['roc_auc']):.3f})", alpha=0.8)
    ax2.plot(pretrained_epochs, pretrained_hist['roc_auc'], 'r-s', linewidth=2, markersize=6,
             label=f"Pretrained (Best: {max(pretrained_hist['roc_auc']):.3f})", alpha=0.8)

    # Add horizontal lines for best values
    ax2.axhline(y=max(scratch_hist['roc_auc']), color='b', linestyle='--', alpha=0.4)
    ax2.axhline(y=max(pretrained_hist['roc_auc']), color='r', linestyle='--', alpha=0.4)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('ROC-AUC', fontsize=12)
    ax2.set_title('Anomaly Detection Performance (ROC-AUC)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.5, max(len(scratch_epochs), len(pretrained_epochs)) + 0.5)
    ax2.set_ylim(0.55, 0.80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'training_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved training comparison to {OUTPUT_DIR / 'training_comparison.png'}")

    # Print summary stats
    print("\n=== Training Summary ===")
    print(f"Scratch UNet:")
    print(f"  Best ROC-AUC: {max(scratch_hist['roc_auc']):.4f} (epoch {scratch_hist['roc_auc'].index(max(scratch_hist['roc_auc']))+1})")
    print(f"  Final Val Loss: {scratch_hist['val_loss'][-1]:.4f}")
    print(f"Pretrained:")
    print(f"  Best ROC-AUC: {max(pretrained_hist['roc_auc']):.4f} (epoch {pretrained_hist['roc_auc'].index(max(pretrained_hist['roc_auc']))+1})")
    print(f"  Final Val Loss: {pretrained_hist['val_loss'][-1]:.4f}")


# ================================================================================
# 2. Image Loading Functions - MUST crop image and mask the same way!
# ================================================================================

def load_jpg_ct_and_mask(image_path, mask_path, target_size=256):
    """Load CT image and mask with ALIGNED cropping.

    The key insight: we must crop both image and mask using the SAME crop coordinates
    derived from the image. Otherwise they will be misaligned.
    """
    # Load image
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img).astype(np.float32) / 255.0

    # Load mask
    mask_img = Image.open(mask_path).convert('L')
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0

    # Get crop coordinates from image
    threshold = 0.05
    margin = 5
    row_sums = np.sum(img_arr > threshold, axis=1)
    col_sums = np.sum(img_arr > threshold, axis=0)
    rows_with_content = np.where(row_sums > 0)[0]
    cols_with_content = np.where(col_sums > 0)[0]

    if len(rows_with_content) > 0 and len(cols_with_content) > 0:
        y_min = max(0, rows_with_content[0] - margin)
        y_max = min(img_arr.shape[0], rows_with_content[-1] + margin)
        x_min = max(0, cols_with_content[0] - margin)
        x_max = min(img_arr.shape[1], cols_with_content[-1] + margin)

        # Crop both with same coordinates
        img_cropped = img_arr[y_min:y_max, x_min:x_max]
        mask_cropped = mask_arr[y_min:y_max, x_min:x_max]
    else:
        img_cropped = img_arr
        mask_cropped = mask_arr

    # Resize image
    img_pil = Image.fromarray((img_cropped * 255).astype(np.uint8))
    img_pil = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    final_img = np.array(img_pil).astype(np.float32) / 255.0
    final_img = final_img * 2.0 - 1.0  # Normalize to [-1, 1]

    # Resize mask with NEAREST to preserve binary values
    mask_pil = Image.fromarray((mask_cropped * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((target_size, target_size), Image.Resampling.NEAREST)
    final_mask = np.array(mask_pil).astype(np.float32) / 255.0
    final_mask = (final_mask > 0.5).astype(np.float32)

    return final_img, final_mask


def load_segmentation_dataset():
    """Load CT ICH segmentation dataset with masks."""
    samples = []
    patients_dir = MASK_DIR / "Patients_CT"

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


# ================================================================================
# 3. Reconstruction Functions
# ================================================================================

@torch.no_grad()
def reconstruct_patched(model, image_tensor, scheduler, device,
                        patch_size=64, stride=32, noise_level=150, is_pretrained=False):
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
            output = model(patched_input, t_tensor)
            if is_pretrained:
                output = output.sample
            pred_x0 = torch.clamp(output, -1, 1)

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


# ================================================================================
# 4. Hemorrhage Visualization
# ================================================================================

def create_hemorrhage_visualization(model_scratch, model_pretrained, scheduler, device):
    """Create visualization comparing models on hemorrhage cases with ground truth masks."""

    # Load segmentation dataset
    samples = load_segmentation_dataset()

    if len(samples) == 0:
        print("No samples found!")
        return

    # Process all samples with scratch model to find best Dice
    print("Processing all samples to find best/random cases...")
    results = []

    for sample in tqdm(samples):
        try:
            # Use aligned loading to ensure mask and image have same cropping
            image, mask = load_jpg_ct_and_mask(sample['image'], sample['mask'], 256)

            # Skip if mask is too small
            if np.sum(mask) < 50:
                continue

            # For scratch model (1 channel)
            image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
            _, error_map = reconstruct_patched(model_scratch, image_tensor, scheduler, device,
                                                is_pretrained=False)
            error_np = error_map.cpu().numpy()

            # Find best threshold for Dice
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
                'dice': best_dice,
                'threshold': best_threshold
            })
        except Exception as e:
            print(f"Error processing {sample['patient']}/{sample['slice']}: {e}")
            continue

    if len(results) < 2:
        print(f"Only found {len(results)} valid results")
        return

    # Sort by Dice score
    results.sort(key=lambda x: x['dice'], reverse=True)

    # Get best and random samples
    best_result = results[0]
    random_result = random.choice(results[len(results)//4:len(results)*3//4])

    print(f"\nBest Dice: {best_result['dice']:.4f}")
    print(f"Random Dice: {random_result['dice']:.4f}")
    print(f"Mean Dice: {np.mean([r['dice'] for r in results]):.4f}")

    # Create visualization for both models
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for row, (result, title_prefix) in enumerate([(best_result, "Best"), (random_result, "Random")]):
        image = result['image']
        mask = result['mask']

        # Process with scratch model (1 channel)
        image_tensor_1ch = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
        recon_scratch, error_scratch = reconstruct_patched(
            model_scratch, image_tensor_1ch, scheduler, device, is_pretrained=False
        )

        # Process with pretrained model (3 channels)
        image_3ch = np.stack([image, image, image], axis=0)
        image_tensor_3ch = torch.tensor(image_3ch).unsqueeze(0).float().to(device)
        recon_pretrained, error_pretrained = reconstruct_patched(
            model_pretrained, image_tensor_3ch, scheduler, device, is_pretrained=True
        )

        # Convert to numpy for display
        img_display = (image + 1) / 2
        recon_scratch_np = (recon_scratch.cpu().numpy() + 1) / 2
        recon_pretrained_np = (recon_pretrained[0].cpu().numpy() + 1) / 2
        error_scratch_np = error_scratch.cpu().numpy()
        error_pretrained_np = error_pretrained.mean(dim=0).cpu().numpy()

        # Col 0: Original
        axes[row, 0].imshow(img_display, cmap='gray')
        axes[row, 0].set_title(f'{title_prefix} (Dice={result["dice"]:.3f})\nOriginal', fontsize=11)
        axes[row, 0].axis('off')

        # Col 1: Ground truth mask
        axes[row, 1].imshow(mask, cmap='Reds')
        axes[row, 1].set_title('GT Hemorrhage Mask', fontsize=11)
        axes[row, 1].axis('off')

        # Col 2: Scratch reconstruction
        axes[row, 2].imshow(recon_scratch_np, cmap='gray')
        axes[row, 2].set_title('Scratch Reconstruction', fontsize=11)
        axes[row, 2].axis('off')

        # Col 3: Scratch error map
        axes[row, 3].imshow(error_scratch_np, cmap='hot')
        axes[row, 3].set_title('Scratch Error Map', fontsize=11)
        axes[row, 3].axis('off')

        # Col 4: Overlay with GT contour
        axes[row, 4].imshow(img_display, cmap='gray')
        axes[row, 4].imshow(error_scratch_np, cmap='hot', alpha=0.5)
        axes[row, 4].contour(mask, colors='lime', linewidths=2)
        axes[row, 4].set_title('Overlay (green=GT)', fontsize=11)
        axes[row, 4].axis('off')

    plt.suptitle(f'pDDPM Hemorrhage Detection (Scratch Model)\nMean Dice: {np.mean([r["dice"] for r in results]):.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hemorrhage_visualization.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'hemorrhage_visualization.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved hemorrhage visualization to {OUTPUT_DIR / 'hemorrhage_visualization.png'}")

    # Create detailed 2x6 comparison figure
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for row, (result, title_prefix) in enumerate([(best_result, "Best"), (random_result, "Random")]):
        image = result['image']
        mask = result['mask']

        # Process with both models
        image_tensor_1ch = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
        recon_scratch, error_scratch = reconstruct_patched(
            model_scratch, image_tensor_1ch, scheduler, device, is_pretrained=False
        )

        image_3ch = np.stack([image, image, image], axis=0)
        image_tensor_3ch = torch.tensor(image_3ch).unsqueeze(0).float().to(device)
        recon_pretrained, error_pretrained = reconstruct_patched(
            model_pretrained, image_tensor_3ch, scheduler, device, is_pretrained=True
        )

        img_display = (image + 1) / 2
        recon_scratch_np = (recon_scratch.cpu().numpy() + 1) / 2
        recon_pretrained_np = (recon_pretrained[0].cpu().numpy() + 1) / 2
        error_scratch_np = error_scratch.cpu().numpy()
        error_pretrained_np = error_pretrained.mean(dim=0).cpu().numpy()

        # Col 0: Original
        axes[row, 0].imshow(img_display, cmap='gray')
        axes[row, 0].set_title(f'{title_prefix}\nOriginal', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')

        # Col 1: GT Mask overlay
        axes[row, 1].imshow(img_display, cmap='gray')
        axes[row, 1].contour(mask, colors='lime', linewidths=2)
        axes[row, 1].set_title('GT Hemorrhage\n(green outline)', fontsize=10)
        axes[row, 1].axis('off')

        # Col 2: Scratch reconstruction
        axes[row, 2].imshow(recon_scratch_np, cmap='gray')
        axes[row, 2].set_title('Scratch\nReconstruction', fontsize=10)
        axes[row, 2].axis('off')

        # Col 3: Scratch error overlay
        axes[row, 3].imshow(img_display, cmap='gray')
        axes[row, 3].imshow(error_scratch_np, cmap='hot', alpha=0.5)
        axes[row, 3].contour(mask, colors='lime', linewidths=2)
        axes[row, 3].set_title('Scratch Error\n+ GT outline', fontsize=10)
        axes[row, 3].axis('off')

        # Col 4: Pretrained reconstruction
        axes[row, 4].imshow(recon_pretrained_np, cmap='gray')
        axes[row, 4].set_title('Pretrained\nReconstruction', fontsize=10)
        axes[row, 4].axis('off')

        # Col 5: Pretrained error overlay
        axes[row, 5].imshow(img_display, cmap='gray')
        axes[row, 5].imshow(error_pretrained_np, cmap='hot', alpha=0.5)
        axes[row, 5].contour(mask, colors='lime', linewidths=2)
        axes[row, 5].set_title('Pretrained Error\n+ GT outline', fontsize=10)
        axes[row, 5].axis('off')

    plt.suptitle('Hemorrhage Detection: Scratch vs Pretrained UNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hemorrhage_visualization_detailed.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'hemorrhage_visualization_detailed.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved detailed visualization to {OUTPUT_DIR / 'hemorrhage_visualization_detailed.png'}")


# ================================================================================
# 5. Reconstruction Examples from RSNA Dataset (Good/Bad, Healthy/Hemorrhage)
# ================================================================================

def create_rsna_reconstruction_examples(model, scheduler, device):
    """Create visualization showing good and bad reconstruction examples
    from healthy and hemorrhage cases in RSNA dataset."""
    import pydicom
    import pandas as pd

    # Load RSNA data
    rsna_dir = DATA_DIR / "rsna-intracranial-hemorrhage-detection"
    dicom_dir = rsna_dir / "stage_2_train"
    csv_path = rsna_dir / "stage_2_train.csv"

    # Get healthy and hemorrhage UIDs
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy_uids = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage_uids = pivot[pivot['any'] == 1].index.tolist()

    # Filter to existing files
    healthy_uids = [u for u in healthy_uids if (dicom_dir / f"ID_{u}.dcm").exists()][:200]
    hemorrhage_uids = [u for u in hemorrhage_uids if (dicom_dir / f"ID_{u}.dcm").exists()][:200]

    print(f"Processing {len(healthy_uids)} healthy and {len(hemorrhage_uids)} hemorrhage samples...")

    def load_dicom(uid):
        """Load and preprocess DICOM file."""
        filepath = dicom_dir / f"ID_{uid}.dcm"
        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array.astype(np.float32)

        # Apply windowing
        intercept = float(dcm.RescaleIntercept) if hasattr(dcm, 'RescaleIntercept') else 0
        slope = float(dcm.RescaleSlope) if hasattr(dcm, 'RescaleSlope') else 1
        img = img * slope + intercept

        # Brain window
        window_center, window_width = 40, 80
        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)

        # Crop brain region
        img = crop_brain_region(img, target_size=256)

        # Normalize to [-1, 1]
        img = img * 2 - 1
        return img

    def get_reconstruction_error(uid):
        """Get reconstruction error score for a UID."""
        try:
            image = load_dicom(uid)
            image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
            _, error_map = reconstruct_patched(model, image_tensor, scheduler, device, is_pretrained=False)
            error = error_map.cpu().numpy().mean()
            return image, error_map.cpu().numpy(), error
        except Exception as e:
            return None, None, None

    # Process samples and collect results
    healthy_results = []
    hemorrhage_results = []

    print("Processing healthy samples...")
    for uid in tqdm(healthy_uids):
        img, error_map, score = get_reconstruction_error(uid)
        if img is not None:
            healthy_results.append({'uid': uid, 'image': img, 'error_map': error_map, 'score': score})

    print("Processing hemorrhage samples...")
    for uid in tqdm(hemorrhage_uids):
        img, error_map, score = get_reconstruction_error(uid)
        if img is not None:
            hemorrhage_results.append({'uid': uid, 'image': img, 'error_map': error_map, 'score': score})

    # Sort by score
    healthy_results.sort(key=lambda x: x['score'])
    hemorrhage_results.sort(key=lambda x: x['score'])

    # Get examples:
    # Good healthy = low error (correctly reconstructed)
    # Bad healthy = high error (false positive)
    # Good hemorrhage = high error (correctly detected)
    # Bad hemorrhage = low error (false negative / missed)

    good_healthy = healthy_results[0]  # Lowest error
    bad_healthy = healthy_results[-1]   # Highest error (false positive)
    good_hemorrhage = hemorrhage_results[-1]  # Highest error (detected)
    bad_hemorrhage = hemorrhage_results[0]    # Lowest error (missed)

    print(f"\nSelected examples:")
    print(f"  Good Healthy (low error): {good_healthy['score']:.4f}")
    print(f"  Bad Healthy (high error - false positive): {bad_healthy['score']:.4f}")
    print(f"  Good Hemorrhage (high error - detected): {good_hemorrhage['score']:.4f}")
    print(f"  Bad Hemorrhage (low error - missed): {bad_hemorrhage['score']:.4f}")

    # Create visualization - 4 rows: Good Healthy, Bad Healthy, Good Hemorrhage, Bad Hemorrhage
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    examples = [
        (good_healthy, "Good Healthy\n(Low Error)", "green"),
        (bad_healthy, "Bad Healthy\n(High Error - FP)", "orange"),
        (good_hemorrhage, "Good Hemorrhage\n(High Error - Detected)", "green"),
        (bad_hemorrhage, "Bad Hemorrhage\n(Low Error - Missed)", "red"),
    ]

    for row, (result, title, color) in enumerate(examples):
        img_display = (result['image'] + 1) / 2
        error_map = result['error_map']

        # Col 0: Original
        axes[row, 0].imshow(img_display, cmap='gray')
        axes[row, 0].set_title(f'{title}\nOriginal', fontsize=10, fontweight='bold', color=color)
        axes[row, 0].axis('off')

        # Col 1: Error map
        im = axes[row, 1].imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
        axes[row, 1].set_title(f'Error Map\n(score: {result["score"]:.4f})', fontsize=10)
        axes[row, 1].axis('off')

        # Col 2: Overlay
        axes[row, 2].imshow(img_display, cmap='gray')
        axes[row, 2].imshow(error_map, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
        axes[row, 2].set_title('Overlay', fontsize=10)
        axes[row, 2].axis('off')

    plt.suptitle('pDDPM Reconstruction: Good vs Bad Examples\n(Scratch UNet on RSNA Dataset)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rsna_reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'rsna_reconstruction_examples.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved RSNA reconstruction examples to {OUTPUT_DIR / 'rsna_reconstruction_examples.png'}")

    # Also create a simpler 2x3 figure showing just one good and one bad for each class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 0: Good examples (correct behavior)
    # Row 1: Bad examples (incorrect behavior)

    row_examples = [
        [(good_healthy, "Healthy\n(Correct: Low Error)"), (good_hemorrhage, "Hemorrhage\n(Correct: High Error)")],
        [(bad_healthy, "Healthy\n(Wrong: High Error)"), (bad_hemorrhage, "Hemorrhage\n(Wrong: Low Error)")]
    ]

    for row, row_data in enumerate(row_examples):
        for col_offset, (result, title) in enumerate(row_data):
            base_col = col_offset * 2
            img_display = (result['image'] + 1) / 2
            error_map = result['error_map']

            # Original
            axes[row, base_col].imshow(img_display, cmap='gray')
            axes[row, base_col].set_title(f'{title}\nOriginal', fontsize=10)
            axes[row, base_col].axis('off')

            # Overlay
            axes[row, base_col + 1].imshow(img_display, cmap='gray')
            axes[row, base_col + 1].imshow(error_map, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
            axes[row, base_col + 1].set_title(f'Error: {result["score"]:.4f}', fontsize=10)
            axes[row, base_col + 1].axis('off')

    axes[0, 0].set_ylabel('Good\n(Correct)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Bad\n(Errors)', fontsize=12, fontweight='bold')

    plt.suptitle('pDDPM Anomaly Detection: Success vs Failure Cases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rsna_good_bad_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'rsna_good_bad_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved good/bad comparison to {OUTPUT_DIR / 'rsna_good_bad_comparison.png'}")


# ================================================================================
# Main
# ================================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create training curves
    print("\n=== Creating Training Curves ===")
    create_training_curves()

    # 2. Load models for visualization
    print("\n=== Loading Models ===")

    # Load scratch model
    model_scratch = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    checkpoint = torch.load(SCRATCH_DIR / "best_model.pt", map_location=device, weights_only=False)
    model_scratch.load_state_dict(checkpoint['model_state_dict'])
    model_scratch.eval()
    print(f"Loaded scratch model (ROC-AUC: {checkpoint.get('roc_auc', 'N/A')})")

    # Load pretrained model
    from diffusers import UNet2DModel
    model_pretrained = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to(device)
    checkpoint = torch.load(PRETRAINED_DIR / "best_model.pt", map_location=device, weights_only=False)
    model_pretrained.load_state_dict(checkpoint['model_state_dict'])
    model_pretrained.eval()
    print(f"Loaded pretrained model (ROC-AUC: {checkpoint.get('roc_auc', 'N/A')})")

    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)

    # 3. Create hemorrhage visualizations
    print("\n=== Creating Hemorrhage Visualizations ===")
    create_hemorrhage_visualization(model_scratch, model_pretrained, scheduler, device)

    # 4. Create RSNA reconstruction examples (good/bad for healthy/hemorrhage)
    print("\n=== Creating RSNA Reconstruction Examples ===")
    create_rsna_reconstruction_examples(model_scratch, scheduler, device)

    print(f"\n=== All figures saved to {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
