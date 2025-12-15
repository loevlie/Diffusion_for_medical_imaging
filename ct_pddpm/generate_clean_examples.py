"""
Generate clean presentation examples with edge-masked error maps.
Removes edge artifacts by masking errors to interior brain region only.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pydicom
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, binary_erosion

from train_patched import UNet, NoiseScheduler, crop_brain_region

# Configuration
DATA_DIR = Path("/media/M2SSD/gen_models_data")
DICOM_DIR = DATA_DIR / "stage_2_train"  # DICOMs are directly in DATA_DIR/stage_2_train
OUTPUT_DIR = Path("presentation_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained scratch model."""
    model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    checkpoint = torch.load("checkpoints_patched/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_dicom(uid, target_size=256):
    """Load and preprocess DICOM file."""
    from PIL import Image

    filepath = DICOM_DIR / f"ID_{uid}.dcm"
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

    # Crop brain region (returns cropped array, not resized)
    img = crop_brain_region(img)

    # Resize to target size
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.array(img_pil).astype(np.float32) / 255.0

    # Normalize to [-1, 1]
    img = img * 2 - 1
    return img


@torch.no_grad()
def reconstruct_patched(model, image_tensor, scheduler,
                        patch_size=64, stride=32, noise_level=150):
    """Reconstruct image using patched approach."""
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
            pred_x0 = torch.clamp(output, -1, 1)

            reconstructed[:, :, y:y+patch_size, x:x+patch_size] += pred_x0[:, :, y:y+patch_size, x:x+patch_size]
            count_map[:, :, y:y+patch_size, x:x+patch_size] += 1

    reconstructed = reconstructed / torch.clamp(count_map, min=1)
    error_map = torch.abs(reconstructed - image_tensor)

    return reconstructed.squeeze(), error_map.squeeze()


def create_brain_interior_mask(image, erosion_iterations=15):
    """
    Create a mask for the interior brain region, excluding edges and skull.

    Args:
        image: normalized image in range [-1, 1] or [0, 1]
        erosion_iterations: how many pixels to erode from the brain boundary
    """
    # Convert to [0, 1] if needed
    if image.min() < 0:
        img = (image + 1) / 2
    else:
        img = image.copy()

    # Create initial brain mask (non-black regions)
    brain_mask = img > 0.05

    # Erode significantly to get interior only (removes skull and edges)
    interior_mask = binary_erosion(brain_mask, iterations=erosion_iterations)

    return interior_mask.astype(np.float32)


def process_error_map(error_map, image, threshold_percentile=95, erosion=15):
    """
    Process error map to show only significant interior errors.

    1. Create interior brain mask (exclude edges/skull)
    2. Apply mask to error map
    3. Threshold to keep only top errors
    4. Smooth slightly for visual appeal
    """
    # Create interior mask
    interior_mask = create_brain_interior_mask(image, erosion_iterations=erosion)

    # Mask the error map to interior only
    masked_error = error_map * interior_mask

    # Smooth slightly
    smoothed = gaussian_filter(masked_error, sigma=1.5)

    # Threshold - keep only top errors within the masked region
    nonzero_errors = smoothed[interior_mask > 0]
    if len(nonzero_errors) > 0:
        threshold = np.percentile(nonzero_errors, threshold_percentile)
        thresholded = np.where(smoothed > threshold, smoothed, 0)
    else:
        thresholded = smoothed

    return thresholded, interior_mask


def visualize_example(image, recon, error_raw, title, output_name, score):
    """Create a 4-panel visualization with clean error map."""

    # Process error map
    error_clean, interior_mask = process_error_map(error_raw, image,
                                                    threshold_percentile=95,
                                                    erosion=15)

    img_display = (image + 1) / 2
    recon_display = (recon + 1) / 2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title(f'Original ({title})', fontsize=12)
    axes[0].axis('off')

    # Reconstruction
    axes[1].imshow(recon_display, cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=12)
    axes[1].axis('off')

    # Error map (clean, interior only)
    axes[2].imshow(error_clean, cmap='hot')
    axes[2].set_title(f'Error Map (top 5%)\nscore: {score:.4f}', fontsize=12)
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(img_display, cmap='gray')
    # Only show overlay where there's actual error
    alpha_mask = np.where(error_clean > 0, 0.7, 0)
    overlay = np.zeros((*error_clean.shape, 4))
    # Red color for errors
    overlay[..., 0] = 1.0  # R
    overlay[..., 3] = alpha_mask  # A
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (top 5% errors)', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {output_name}")


def main():
    print(f"Using device: {device}")

    # Load model and scheduler
    model = load_model()
    scheduler = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)

    # Load RSNA labels - CSV is in DATA_DIR, not RSNA_DIR
    csv_path = DATA_DIR / "stage_2_train.csv"
    print(f"Loading labels from {csv_path}")
    df = pd.read_csv(csv_path)
    df['sop_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    df = df.drop_duplicates(subset=['sop_uid', 'hemorrhage_type'])
    pivot = df.pivot(index='sop_uid', columns='hemorrhage_type', values='Label')

    healthy_uids = pivot[pivot['any'] == 0].index.tolist()
    hemorrhage_uids = pivot[pivot['any'] == 1].index.tolist()

    healthy_uids = [u for u in healthy_uids if (DICOM_DIR / f"ID_{u}.dcm").exists()][:100]
    hemorrhage_uids = [u for u in hemorrhage_uids if (DICOM_DIR / f"ID_{u}.dcm").exists()][:100]

    print(f"Processing {len(healthy_uids)} healthy, {len(hemorrhage_uids)} hemorrhage samples...")

    # Process and collect results
    healthy_results = []
    hemorrhage_results = []

    errors_seen = []
    for uid in tqdm(healthy_uids, desc="Healthy"):
        try:
            image = load_dicom(uid)
            image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
            recon, error_map = reconstruct_patched(model, image_tensor, scheduler)
            error_np = error_map.cpu().numpy()
            score = error_np.mean()
            healthy_results.append({
                'uid': uid, 'image': image,
                'recon': recon.cpu().numpy(),
                'error_map': error_np, 'score': score
            })
        except Exception as e:
            if str(e) not in errors_seen:
                print(f"Error: {e}")
                errors_seen.append(str(e))
            continue

    for uid in tqdm(hemorrhage_uids, desc="Hemorrhage"):
        try:
            image = load_dicom(uid)
            image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
            recon, error_map = reconstruct_patched(model, image_tensor, scheduler)
            error_np = error_map.cpu().numpy()
            score = error_np.mean()
            hemorrhage_results.append({
                'uid': uid, 'image': image,
                'recon': recon.cpu().numpy(),
                'error_map': error_np, 'score': score
            })
        except Exception as e:
            continue

    # Sort by score
    healthy_results.sort(key=lambda x: x['score'])
    hemorrhage_results.sort(key=lambda x: x['score'])

    print(f"\nFound {len(healthy_results)} healthy, {len(hemorrhage_results)} hemorrhage results")

    # Generate clean examples
    # Good healthy = low error (correctly reconstructed)
    # False positive healthy = high error (incorrectly flagged)
    # Detected hemorrhage = high error (correctly detected)
    # Missed hemorrhage = low error (incorrectly missed)

    # Select examples
    good_healthy = healthy_results[0]
    fp_healthy = healthy_results[-1]
    detected_hem = hemorrhage_results[-1]
    missed_hem = hemorrhage_results[0]

    # Also get some mid-range healthy examples
    mid_idx = len(healthy_results) // 2
    healthy_mid_1 = healthy_results[mid_idx]
    healthy_mid_2 = healthy_results[mid_idx + 1]

    print(f"\nSelected examples:")
    print(f"  Good healthy score: {good_healthy['score']:.4f}")
    print(f"  FP healthy score: {fp_healthy['score']:.4f}")
    print(f"  Detected hemorrhage score: {detected_hem['score']:.4f}")
    print(f"  Missed hemorrhage score: {missed_hem['score']:.4f}")

    # Generate visualizations
    visualize_example(good_healthy['image'], good_healthy['recon'],
                      good_healthy['error_map'], "Healthy",
                      "clean_healthy_1.png", good_healthy['score'])

    visualize_example(healthy_mid_1['image'], healthy_mid_1['recon'],
                      healthy_mid_1['error_map'], "Healthy",
                      "clean_healthy_2.png", healthy_mid_1['score'])

    visualize_example(detected_hem['image'], detected_hem['recon'],
                      detected_hem['error_map'], "Hemorrhage",
                      "clean_hemorrhage_detected.png", detected_hem['score'])

    visualize_example(missed_hem['image'], missed_hem['recon'],
                      missed_hem['error_map'], "Hemorrhage",
                      "clean_hemorrhage_missed.png", missed_hem['score'])

    print(f"\nAll clean examples saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
