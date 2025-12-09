#!/usr/bin/env python3
"""
Create clear visualization of hemorrhage with proper annotation.
Specifically look for large, obvious bleeds and annotate them clearly.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
import pydicom
from pathlib import Path


def load_dicom_image(dcm_path):
    """Load DICOM and return raw pixel array."""
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply rescale if available
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept

    return img


def apply_window(img, window_center, window_width):
    """Apply CT window to get proper visualization."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_windowed = np.clip(img, img_min, img_max)
    img_windowed = (img_windowed - img_min) / (img_max - img_min)
    return img_windowed


def find_hemorrhage_region(img_hu):
    """
    Find the hemorrhage region in a CT scan.
    Blood appears at HU 50-100 (acute hemorrhage).
    Returns mask and centroid of largest blood region.
    """
    from scipy import ndimage

    # Blood window: HU 50-100 for acute blood
    # But we need to exclude bone (>100 HU) and be within brain
    blood_mask = (img_hu > 50) & (img_hu < 100)

    # Also create brain mask (exclude air and bone)
    brain_mask = (img_hu > 0) & (img_hu < 80)

    # For hemorrhage, look for bright areas that are NOT bone
    # Subdural/epidural appear along skull edge
    # Intraparenchymal appears within brain tissue
    hemorrhage_mask = (img_hu > 55) & (img_hu < 90)

    # Label connected components
    labeled, num_features = ndimage.label(hemorrhage_mask)

    if num_features == 0:
        return None, None, None

    # Find the largest component (likely the hemorrhage)
    component_sizes = ndimage.sum(hemorrhage_mask, labeled, range(1, num_features + 1))
    largest_idx = np.argmax(component_sizes) + 1
    largest_mask = labeled == largest_idx

    # Get centroid
    cy, cx = ndimage.center_of_mass(largest_mask)

    return hemorrhage_mask, (int(cx), int(cy)), component_sizes[largest_idx - 1]


def main():
    data_dir = Path("/media/M2SSD/gen_models_data")
    labels_path = data_dir / "stage_2_train.csv"
    dicom_dir = data_dir / "stage_2_train"

    print("Loading labels...")
    df = pd.read_csv(labels_path)

    # Pivot to get one row per image
    df_pivot = df.pivot(index='ID', columns='Label', values='Label')
    df_pivot = df_pivot.reset_index()

    # Extract image ID
    df_pivot['image_id'] = df_pivot['ID'].apply(lambda x: x.replace('_epidural', '').replace('_intraparenchymal', '')
                                                  .replace('_intraventricular', '').replace('_subarachnoid', '')
                                                  .replace('_subdural', '').replace('_any', ''))

    # Get unique images with hemorrhage labels
    df_labels = df.copy()
    df_labels['image_id'] = df_labels['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df_labels['hemorrhage_type'] = df_labels['ID'].apply(lambda x: '_'.join(x.split('_')[2:]))

    # Pivot to one row per image
    df_wide = df_labels.pivot(index='image_id', columns='hemorrhage_type', values='Label').reset_index()

    # Find images with specific hemorrhage types
    # Intraparenchymal and epidural tend to be more visible
    print("\nLooking for cases with large, visible hemorrhages...")

    hemorrhage_types = ['epidural', 'intraparenchymal', 'subdural', 'intraventricular', 'subarachnoid']

    # Get images with hemorrhage
    hemorrhage_images = df_wide[df_wide['any'] == 1]['image_id'].tolist()
    healthy_images = df_wide[df_wide['any'] == 0]['image_id'].tolist()

    print(f"Found {len(hemorrhage_images)} hemorrhage images")
    print(f"Found {len(healthy_images)} healthy images")

    # Search for clearly visible hemorrhages by checking image characteristics
    best_hemorrhage_cases = []

    print("\nScanning for clearly visible hemorrhages...")
    np.random.seed(42)
    sample_hemorrhage = np.random.choice(hemorrhage_images, min(500, len(hemorrhage_images)), replace=False)

    for img_id in sample_hemorrhage:
        dcm_path = dicom_dir / f"{img_id}.dcm"
        if not dcm_path.exists():
            continue

        try:
            img_hu = load_dicom_image(dcm_path)

            # Look for large bright regions (hemorrhage)
            mask, centroid, size = find_hemorrhage_region(img_hu)

            if size is not None and size > 500:  # Significant hemorrhage size
                # Get hemorrhage type for this image
                row = df_wide[df_wide['image_id'] == img_id].iloc[0]
                hem_types = [t for t in hemorrhage_types if row.get(t, 0) == 1]

                best_hemorrhage_cases.append({
                    'image_id': img_id,
                    'size': size,
                    'centroid': centroid,
                    'types': hem_types
                })
        except Exception as e:
            continue

    # Sort by hemorrhage size
    best_hemorrhage_cases.sort(key=lambda x: x['size'], reverse=True)

    print(f"\nFound {len(best_hemorrhage_cases)} cases with visible hemorrhage")
    if best_hemorrhage_cases:
        print("Top 5 by size:")
        for case in best_hemorrhage_cases[:5]:
            print(f"  {case['image_id']}: size={case['size']:.0f}, types={case['types']}")

    # Create visualization with the best case
    if not best_hemorrhage_cases:
        print("No clear hemorrhage cases found!")
        return

    best_case = best_hemorrhage_cases[0]

    # Also find a healthy image at similar brain level
    best_hemorrhage_img = load_dicom_image(dicom_dir / f"{best_case['image_id']}.dcm")

    # Feature to match: amount of ventricle (dark CSF regions)
    def get_brain_level_features(img_hu):
        """Get features to match brain level."""
        # Brain mask
        brain = (img_hu > 0) & (img_hu < 80)
        brain_area = np.sum(brain)

        # CSF (ventricles) - very dark
        csf = (img_hu > 0) & (img_hu < 20)
        csf_ratio = np.sum(csf) / max(brain_area, 1)

        # Gray/white matter ratio
        gray = (img_hu > 30) & (img_hu < 45)
        white = (img_hu > 20) & (img_hu < 35)

        return {'brain_area': brain_area, 'csf_ratio': csf_ratio}

    target_features = get_brain_level_features(best_hemorrhage_img)

    # Find matching healthy image
    print("\nFinding matching healthy brain level...")
    best_match = None
    best_match_score = float('inf')

    sample_healthy = np.random.choice(healthy_images, min(200, len(healthy_images)), replace=False)

    for img_id in sample_healthy:
        dcm_path = dicom_dir / f"{img_id}.dcm"
        if not dcm_path.exists():
            continue

        try:
            img_hu = load_dicom_image(dcm_path)
            features = get_brain_level_features(img_hu)

            # Score based on feature similarity
            score = abs(features['brain_area'] - target_features['brain_area']) / 1000
            score += abs(features['csf_ratio'] - target_features['csf_ratio']) * 100

            if score < best_match_score:
                best_match_score = score
                best_match = img_id
        except:
            continue

    print(f"Best matching healthy image: {best_match}")

    # Load both images
    hemorrhage_img = best_hemorrhage_img
    healthy_img = load_dicom_image(dicom_dir / f"{best_match}.dcm")

    # Apply brain window
    hemorrhage_windowed = apply_window(hemorrhage_img, 40, 80)
    healthy_windowed = apply_window(healthy_img, 40, 80)

    # Find hemorrhage location again
    mask, centroid, _ = find_hemorrhage_region(hemorrhage_img)

    # Create the visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Intracranial Hemorrhage Detection\nCT Brain Window (W:80, L:40)', fontsize=14, fontweight='bold')

    # Healthy brain
    axes[0].imshow(healthy_windowed, cmap='gray')
    axes[0].set_title('HEALTHY BRAIN', color='green', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Hemorrhage brain
    axes[1].imshow(hemorrhage_windowed, cmap='gray')
    types_str = ', '.join(best_case['types']).upper()
    axes[1].set_title(f'HEMORRHAGE ({types_str})', color='red', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add arrow pointing to hemorrhage
    if centroid:
        cx, cy = centroid
        # Draw arrow from outside pointing to hemorrhage
        arrow_start = (cx + 80, cy - 80)  # Start point (outside)
        arrow_end = (cx + 10, cy - 10)    # End point (near hemorrhage)

        axes[1].annotate('HEMORRHAGE\nHERE',
                        xy=arrow_end, xytext=arrow_start,
                        fontsize=12, color='yellow', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=3),
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

    # Hemorrhage with overlay
    axes[2].imshow(hemorrhage_windowed, cmap='gray')

    # Create better hemorrhage mask - threshold bright areas
    blood_mask = (hemorrhage_img > 55) & (hemorrhage_img < 85)

    # Remove small components
    from scipy import ndimage
    blood_mask = ndimage.binary_opening(blood_mask, iterations=2)
    blood_mask = ndimage.binary_closing(blood_mask, iterations=2)

    # Create red overlay only where blood is detected
    overlay = np.zeros((*hemorrhage_windowed.shape, 4))
    overlay[blood_mask, 0] = 1.0  # Red
    overlay[blood_mask, 3] = 0.6  # Alpha

    axes[2].imshow(overlay)
    axes[2].set_title('BLOOD HIGHLIGHTED (RED OVERLAY)', color='orange', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add text explanation
    fig.text(0.5, 0.02,
             'Blood appears BRIGHT (hyperdense) on CT. Fresh blood has HU values of 50-90.\n'
             'The hemorrhage is highlighted in red in the rightmost image.',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('ct_clear_hemorrhage.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("\nSaved: ct_clear_hemorrhage.png")

    # Create a second visualization with multiple hemorrhage types
    print("\nCreating multi-type hemorrhage visualization...")

    # Find one example of each hemorrhage type
    type_examples = {}
    for hem_type in ['epidural', 'intraparenchymal', 'subdural']:
        for case in best_hemorrhage_cases:
            if hem_type in case['types'] and hem_type not in type_examples:
                type_examples[hem_type] = case
                break

    if len(type_examples) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Different Types of Intracranial Hemorrhage\n(Blood appears BRIGHT/WHITE on CT)',
                     fontsize=16, fontweight='bold', color='white')

        for idx, (hem_type, case) in enumerate(type_examples.items()):
            if idx >= 3:
                break

            img_hu = load_dicom_image(dicom_dir / f"{case['image_id']}.dcm")
            img_windowed = apply_window(img_hu, 40, 80)

            # Top row: original
            axes[0, idx].imshow(img_windowed, cmap='gray')
            axes[0, idx].set_title(f'{hem_type.upper()}', color='red', fontsize=14, fontweight='bold')
            axes[0, idx].axis('off')

            # Add arrow to hemorrhage location
            cx, cy = case['centroid']
            axes[0, idx].annotate('', xy=(cx, cy), xytext=(cx + 60, cy - 60),
                                 arrowprops=dict(arrowstyle='->', color='yellow', lw=3))

            # Bottom row: with blood overlay
            axes[1, idx].imshow(img_windowed, cmap='gray')

            blood_mask = (img_hu > 55) & (img_hu < 85)
            blood_mask = ndimage.binary_opening(blood_mask, iterations=1)

            overlay = np.zeros((*img_windowed.shape, 4))
            overlay[blood_mask, 0] = 1.0
            overlay[blood_mask, 3] = 0.5

            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title('Blood Highlighted', color='orange', fontsize=12)
            axes[1, idx].axis('off')

        # Add descriptions
        descriptions = {
            'epidural': 'EPIDURAL: Blood between skull and dura\n(lens-shaped, limited by sutures)',
            'intraparenchymal': 'INTRAPARENCHYMAL: Blood within brain tissue\n(often from hypertension or trauma)',
            'subdural': 'SUBDURAL: Blood between dura and brain\n(crescent-shaped, crosses sutures)'
        }

        for idx, hem_type in enumerate(type_examples.keys()):
            if idx >= 3:
                break
            axes[1, idx].text(0.5, -0.15, descriptions.get(hem_type, ''),
                             transform=axes[1, idx].transAxes,
                             ha='center', fontsize=10, color='white')

        plt.tight_layout()
        plt.savefig('ct_hemorrhage_types.png', dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()
        print("Saved: ct_hemorrhage_types.png")

    # Create zoomed-in view of hemorrhage
    print("\nCreating zoomed hemorrhage view...")

    best_img = load_dicom_image(dicom_dir / f"{best_case['image_id']}.dcm")
    best_windowed = apply_window(best_img, 40, 80)

    cx, cy = best_case['centroid']

    # Define zoom region around hemorrhage
    zoom_size = 100
    y1 = max(0, cy - zoom_size)
    y2 = min(best_windowed.shape[0], cy + zoom_size)
    x1 = max(0, cx - zoom_size)
    x2 = min(best_windowed.shape[1], cx + zoom_size)

    zoomed = best_windowed[y1:y2, x1:x2]
    zoomed_hu = best_img[y1:y2, x1:x2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Zoomed View of Hemorrhage', fontsize=14, fontweight='bold', color='white')

    # Full image with box
    axes[0].imshow(best_windowed, cmap='gray')
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='yellow', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].set_title('Full CT Scan', color='white', fontsize=12)
    axes[0].axis('off')

    # Zoomed region
    axes[1].imshow(zoomed, cmap='gray')
    axes[1].set_title('Zoomed Region (Brain Window)', color='white', fontsize=12)
    axes[1].axis('off')

    # Zoomed with blood highlighted
    axes[2].imshow(zoomed, cmap='gray')
    blood_mask_zoom = (zoomed_hu > 55) & (zoomed_hu < 85)
    overlay_zoom = np.zeros((*zoomed.shape, 4))
    overlay_zoom[blood_mask_zoom, 0] = 1.0
    overlay_zoom[blood_mask_zoom, 3] = 0.6
    axes[2].imshow(overlay_zoom)
    axes[2].set_title('Blood Highlighted (RED)', color='orange', fontsize=12)
    axes[2].axis('off')

    # Add arrow pointing to the bright region
    axes[1].annotate('BLOOD\n(bright)', xy=(zoom_size, zoom_size), xytext=(zoom_size + 40, zoom_size - 50),
                    fontsize=11, color='yellow', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2))

    plt.tight_layout()
    plt.savefig('ct_hemorrhage_zoomed.png', dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print("Saved: ct_hemorrhage_zoomed.png")

    print("\n=== VISUALIZATION COMPLETE ===")
    print("Created files:")
    print("  - ct_clear_hemorrhage.png : Side-by-side healthy vs hemorrhage with annotation")
    print("  - ct_hemorrhage_types.png : Different hemorrhage types explained")
    print("  - ct_hemorrhage_zoomed.png : Zoomed view of hemorrhage region")


if __name__ == "__main__":
    main()
