#!/usr/bin/env python3
"""
Create multiple examples showing blood (hemorrhage) clearly highlighted.

KEY MEDICAL CONCEPT:
- Hemorrhage = bleeding = blood where it shouldn't be
- In the brain, blood appears BRIGHT (white/light gray) on CT scans
- Fresh blood has Hounsfield Units (HU) of approximately 50-90
- Normal brain tissue is darker (gray matter ~35-40 HU, white matter ~25-30 HU)
- The bright white areas we're highlighting ARE the hemorrhage (blood)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pathlib import Path
from scipy import ndimage


def load_dicom_image(dcm_path):
    """Load DICOM and return raw pixel array in Hounsfield Units."""
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept
    return img


def apply_window(img, window_center, window_width):
    """Apply CT window for visualization."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_windowed = np.clip(img, img_min, img_max)
    img_windowed = (img_windowed - img_min) / (img_max - img_min)
    return img_windowed


def find_blood_regions(img_hu, min_size=200):
    """
    Find blood regions in CT scan.
    Blood (acute hemorrhage) appears at HU 50-90.
    Returns mask of blood regions.
    """
    # Blood detection threshold
    blood_mask = (img_hu > 50) & (img_hu < 90)

    # Clean up noise
    blood_mask = ndimage.binary_opening(blood_mask, iterations=2)
    blood_mask = ndimage.binary_closing(blood_mask, iterations=2)

    # Remove small components
    labeled, num_features = ndimage.label(blood_mask)
    if num_features > 0:
        component_sizes = ndimage.sum(blood_mask, labeled, range(1, num_features + 1))
        # Keep only components larger than min_size
        for i, size in enumerate(component_sizes):
            if size < min_size:
                blood_mask[labeled == (i + 1)] = False

    return blood_mask


def get_blood_amount(img_hu):
    """Calculate amount of blood in image (for finding good examples)."""
    blood_mask = find_blood_regions(img_hu, min_size=100)
    return np.sum(blood_mask)


def main():
    data_dir = Path("/media/M2SSD/gen_models_data")
    labels_path = data_dir / "stage_2_train.csv"
    dicom_dir = data_dir / "stage_2_train"

    print("Loading labels...")
    df = pd.read_csv(labels_path)

    # Get image IDs and their hemorrhage status
    df['image_id'] = df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['hemorrhage_type'] = df['ID'].apply(lambda x: '_'.join(x.split('_')[2:]))
    # Remove duplicates before pivot
    df_dedup = df.drop_duplicates(subset=['image_id', 'hemorrhage_type'])
    df_wide = df_dedup.pivot(index='image_id', columns='hemorrhage_type', values='Label').reset_index()

    hemorrhage_images = df_wide[df_wide['any'] == 1]['image_id'].tolist()
    healthy_images = df_wide[df_wide['any'] == 0]['image_id'].tolist()

    print(f"Hemorrhage images: {len(hemorrhage_images)}")
    print(f"Healthy images: {len(healthy_images)}")

    # Find hemorrhage images with LOTS of visible blood
    print("\nSearching for images with clearly visible blood...")

    np.random.seed(123)
    sample = np.random.choice(hemorrhage_images, min(1000, len(hemorrhage_images)), replace=False)

    blood_amounts = []
    for img_id in sample:
        dcm_path = dicom_dir / f"{img_id}.dcm"
        if not dcm_path.exists():
            continue
        try:
            img_hu = load_dicom_image(dcm_path)
            blood = get_blood_amount(img_hu)
            if blood > 500:  # Significant blood
                # Get hemorrhage types
                row = df_wide[df_wide['image_id'] == img_id].iloc[0]
                types = [t for t in ['epidural', 'intraparenchymal', 'subdural', 'intraventricular', 'subarachnoid']
                        if row.get(t, 0) == 1]
                blood_amounts.append({
                    'image_id': img_id,
                    'blood_amount': blood,
                    'types': types
                })
        except:
            continue

    # Sort by blood amount
    blood_amounts.sort(key=lambda x: x['blood_amount'], reverse=True)

    print(f"Found {len(blood_amounts)} images with significant blood")
    print("\nTop 10 by blood amount:")
    for case in blood_amounts[:10]:
        print(f"  {case['image_id']}: {case['blood_amount']:.0f} pixels, types: {case['types']}")

    # Create visualization with multiple examples
    # ==========================================

    # Figure 1: Multiple hemorrhage examples with blood highlighted
    print("\n\nCreating Figure 1: Multiple hemorrhage examples...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor('black')

    fig.suptitle('INTRACRANIAL HEMORRHAGE = BLOOD IN THE BRAIN\n'
                 'Blood appears BRIGHT (white) on CT scans - highlighted in RED below',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # Use top 6 cases
    for idx, case in enumerate(blood_amounts[:6]):
        row = idx // 2
        col = (idx % 2) * 2

        img_hu = load_dicom_image(dicom_dir / f"{case['image_id']}.dcm")
        img_windowed = apply_window(img_hu, 40, 80)  # Brain window

        # Left: Original CT
        axes[row, col].imshow(img_windowed, cmap='gray')
        axes[row, col].set_title(f"CT Scan\n({', '.join(case['types'][:2])})",
                                 color='white', fontsize=11)
        axes[row, col].axis('off')

        # Right: With blood highlighted
        axes[row, col+1].imshow(img_windowed, cmap='gray')

        blood_mask = find_blood_regions(img_hu, min_size=150)
        overlay = np.zeros((*img_windowed.shape, 4))
        overlay[blood_mask, 0] = 1.0  # Red
        overlay[blood_mask, 3] = 0.7  # Alpha
        axes[row, col+1].imshow(overlay)

        axes[row, col+1].set_title('BLOOD HIGHLIGHTED\n(This IS the hemorrhage)',
                                   color='red', fontsize=11, fontweight='bold')
        axes[row, col+1].axis('off')

    # Add explanation at bottom
    fig.text(0.5, 0.02,
             'HEMORRHAGE means bleeding. On CT, blood appears bright white because it has higher density.\n'
             'The red highlighted areas show exactly where the blood (hemorrhage) is located in the brain.',
             ha='center', fontsize=14, color='yellow', style='italic',
             bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('ct_blood_is_hemorrhage.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Saved: ct_blood_is_hemorrhage.png")

    # Figure 2: Healthy vs Hemorrhage comparison
    print("\nCreating Figure 2: Healthy vs Hemorrhage comparison...")

    # Find healthy images
    np.random.seed(456)
    healthy_sample = np.random.choice(healthy_images, min(100, len(healthy_images)), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('black')

    fig.suptitle('HEALTHY BRAIN vs HEMORRHAGE (Blood in Brain)\n'
                 'Notice: Healthy brains have NO bright white blood spots',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    # Top row: Healthy examples
    for idx, img_id in enumerate(healthy_sample[:4]):
        dcm_path = dicom_dir / f"{img_id}.dcm"
        if not dcm_path.exists():
            continue
        try:
            img_hu = load_dicom_image(dcm_path)
            img_windowed = apply_window(img_hu, 40, 80)

            axes[0, idx].imshow(img_windowed, cmap='gray')
            axes[0, idx].set_title('HEALTHY\n(No blood)', color='green', fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
        except:
            continue

    # Bottom row: Hemorrhage examples with highlighting
    for idx, case in enumerate(blood_amounts[:4]):
        img_hu = load_dicom_image(dicom_dir / f"{case['image_id']}.dcm")
        img_windowed = apply_window(img_hu, 40, 80)

        axes[1, idx].imshow(img_windowed, cmap='gray')

        blood_mask = find_blood_regions(img_hu, min_size=150)
        overlay = np.zeros((*img_windowed.shape, 4))
        overlay[blood_mask, 0] = 1.0
        overlay[blood_mask, 3] = 0.7
        axes[1, idx].imshow(overlay)

        axes[1, idx].set_title('HEMORRHAGE\n(Blood = red areas)', color='red', fontsize=12, fontweight='bold')
        axes[1, idx].axis('off')

    fig.text(0.5, 0.02,
             'The RED highlighted regions show blood (hemorrhage) in the brain.\n'
             'Healthy brains do not have these bright spots.',
             ha='center', fontsize=14, color='yellow',
             bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig('ct_healthy_vs_blood.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Saved: ct_healthy_vs_blood.png")

    # Figure 3: Educational diagram explaining what we see
    print("\nCreating Figure 3: Educational diagram...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('black')

    # Use the case with most blood
    best_case = blood_amounts[0]
    img_hu = load_dicom_image(dicom_dir / f"{best_case['image_id']}.dcm")
    img_windowed = apply_window(img_hu, 40, 80)

    # Panel 1: Original
    axes[0].imshow(img_windowed, cmap='gray')
    axes[0].set_title('Original CT Scan', color='white', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Add annotations
    axes[0].text(0.05, 0.95, 'DARK = Brain tissue\n(normal)', transform=axes[0].transAxes,
                 fontsize=11, color='cyan', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[0].text(0.05, 0.05, 'BRIGHT = Blood\n(hemorrhage!)', transform=axes[0].transAxes,
                 fontsize=11, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Panel 2: With blood mask
    axes[1].imshow(img_windowed, cmap='gray')
    blood_mask = find_blood_regions(img_hu, min_size=150)
    overlay = np.zeros((*img_windowed.shape, 4))
    overlay[blood_mask, 0] = 1.0
    overlay[blood_mask, 3] = 0.8
    axes[1].imshow(overlay)
    axes[1].set_title('Blood (Hemorrhage) Highlighted', color='red', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Just the blood mask
    axes[2].imshow(blood_mask, cmap='Reds')
    axes[2].set_title('Blood Regions Only\n(This is what the model learns to detect)',
                      color='orange', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    fig.suptitle(f'Understanding CT Hemorrhage Detection\nCase: {", ".join(best_case["types"])} hemorrhage',
                 fontsize=16, fontweight='bold', color='white', y=1.02)

    # Explanation
    fig.text(0.5, 0.02,
             'HEMORRHAGE = Blood in the wrong place. On CT scans, blood appears BRIGHT (white).\n'
             'Our pDDPM model learns what healthy brains look like, then detects blood as "anomaly".',
             ha='center', fontsize=13, color='yellow',
             bbox=dict(boxstyle='round', facecolor='navy', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig('ct_hemorrhage_explained.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Saved: ct_hemorrhage_explained.png")

    # Figure 4: Single best example, large and clear
    print("\nCreating Figure 4: Single clear example...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('black')

    best_case = blood_amounts[0]
    img_hu = load_dicom_image(dicom_dir / f"{best_case['image_id']}.dcm")
    img_windowed = apply_window(img_hu, 40, 80)

    # Left: Original
    axes[0].imshow(img_windowed, cmap='gray')
    axes[0].set_title('CT Brain Scan with Hemorrhage', color='white', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Add arrow pointing to blood
    blood_mask = find_blood_regions(img_hu, min_size=150)
    ys, xs = np.where(blood_mask)
    if len(xs) > 0:
        cx, cy = np.mean(xs), np.mean(ys)
        axes[0].annotate('BLOOD\n(Hemorrhage)',
                        xy=(cx, cy), xytext=(cx + 100, cy - 100),
                        fontsize=14, color='yellow', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=3),
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))

    # Right: With overlay
    axes[1].imshow(img_windowed, cmap='gray')
    overlay = np.zeros((*img_windowed.shape, 4))
    overlay[blood_mask, 0] = 1.0
    overlay[blood_mask, 3] = 0.7
    axes[1].imshow(overlay)
    axes[1].set_title('Blood Highlighted in RED', color='red', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    fig.suptitle(f'Intracranial Hemorrhage: {", ".join(best_case["types"]).upper()}\n'
                 'The bright white area IS the blood (hemorrhage)',
                 fontsize=18, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('ct_single_hemorrhage.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Saved: ct_single_hemorrhage.png")

    print("\n" + "="*60)
    print("SUMMARY: What is Hemorrhage?")
    print("="*60)
    print("""
    HEMORRHAGE = BLEEDING = BLOOD in the wrong place

    In brain CT scans:
    - Normal brain tissue appears GRAY (darker)
    - Blood appears WHITE/BRIGHT (higher density)
    - The bright spots we highlight ARE the hemorrhage

    Types of brain hemorrhage:
    - EPIDURAL: Blood between skull and brain covering
    - SUBDURAL: Blood under brain covering
    - INTRAPARENCHYMAL: Blood inside brain tissue
    - INTRAVENTRICULAR: Blood in brain ventricles
    - SUBARACHNOID: Blood in space around brain

    All of these show up as BRIGHT areas on CT!
    """)
    print("="*60)
    print("\nCreated visualizations:")
    print("  - ct_blood_is_hemorrhage.png : Multiple examples with blood highlighted")
    print("  - ct_healthy_vs_blood.png : Comparison of healthy vs hemorrhage")
    print("  - ct_hemorrhage_explained.png : Educational diagram")
    print("  - ct_single_hemorrhage.png : Single clear example with annotation")


if __name__ == "__main__":
    main()
