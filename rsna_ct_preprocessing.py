"""
RSNA Intracranial Hemorrhage CT Preprocessing for Diffusion Model Fine-tuning
==============================================================================

This script prepares CT slices from the RSNA Intracranial Hemorrhage dataset
for fine-tuning FLUX or other diffusion models.

Dataset: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
You'll need to download stage_2_train.csv and the stage_2_train_images folder.

Usage:
    python rsna_ct_preprocessing.py --input_dir /path/to/rsna --output_dir ./ct_dataset --num_samples 1000
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
import random

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CT Windowing Functions
# =============================================================================

def apply_window(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """Apply CT windowing to convert HU values to displayable range."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return windowed


def get_windowed_image(pixel_array: np.ndarray, 
                       intercept: float, 
                       slope: float,
                       window_center: int = 40, 
                       window_width: int = 80) -> np.ndarray:
    """Convert raw DICOM pixels to windowed HU image."""
    # Convert to Hounsfield Units
    hu_image = pixel_array * slope + intercept
    # Apply window
    return apply_window(hu_image, window_center, window_width)


def create_three_channel_ct(pixel_array: np.ndarray,
                            intercept: float,
                            slope: float) -> np.ndarray:
    """
    Create 3-channel image using different windows for brain CT:
    - Red channel: Brain window (W:80, L:40) - for soft tissue
    - Green channel: Subdural window (W:200, L:80) - for blood
    - Blue channel: Bone window (W:2800, L:600) - for skull/calcifications
    
    This preserves more information than a single window.
    """
    brain = get_windowed_image(pixel_array, intercept, slope, 
                               window_center=40, window_width=80)
    subdural = get_windowed_image(pixel_array, intercept, slope, 
                                  window_center=80, window_width=200)
    bone = get_windowed_image(pixel_array, intercept, slope, 
                              window_center=600, window_width=2800)
    
    return np.stack([brain, subdural, bone], axis=-1)


# =============================================================================
# DICOM Loading
# =============================================================================

def load_dicom(filepath: str) -> Tuple[np.ndarray, float, float]:
    """Load DICOM file and extract pixel array with rescale parameters."""
    dcm = pydicom.dcmread(filepath)
    
    # Get rescale parameters (for HU conversion)
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    slope = float(getattr(dcm, 'RescaleSlope', 1))
    
    # Get pixel array
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    return pixel_array, intercept, slope


def get_dicom_metadata(filepath: str) -> dict:
    """Extract relevant metadata from DICOM for captioning."""
    dcm = pydicom.dcmread(filepath)
    
    metadata = {
        'slice_location': getattr(dcm, 'SliceLocation', None),
        'slice_thickness': getattr(dcm, 'SliceThickness', None),
        'patient_position': getattr(dcm, 'PatientPosition', None),
        'image_position': getattr(dcm, 'ImagePositionPatient', None),
    }
    
    return metadata


# =============================================================================
# Caption Generation
# =============================================================================

def generate_caption(hemorrhage_types: list, 
                     include_negative: bool = True) -> str:
    """
    Generate text caption for CT slice based on hemorrhage labels.
    
    Args:
        hemorrhage_types: List of hemorrhage types present in the slice
        include_negative: Whether to include description for normal scans
    """
    base = "axial CT scan of the brain"
    
    if not hemorrhage_types or hemorrhage_types == ['any']:
        if include_negative:
            return f"{base}, normal appearance, no acute intracranial hemorrhage"
        return base
    
    # Map hemorrhage types to descriptive text
    type_descriptions = {
        'epidural': 'epidural hematoma',
        'subdural': 'subdural hematoma', 
        'subarachnoid': 'subarachnoid hemorrhage',
        'intraventricular': 'intraventricular hemorrhage',
        'intraparenchymal': 'intraparenchymal hemorrhage',
    }
    
    findings = [type_descriptions.get(t, t) for t in hemorrhage_types if t != 'any']
    
    if findings:
        findings_str = ', '.join(findings)
        return f"{base} showing {findings_str}"
    
    return base


# =============================================================================
# Dataset Processing
# =============================================================================

def parse_rsna_labels(csv_path: str) -> pd.DataFrame:
    """
    Parse RSNA labels CSV into a more usable format.
    
    The original format has one row per (image, hemorrhage_type) pair.
    We convert to one row per image with all hemorrhage types.
    """
    df = pd.read_csv(csv_path)
    
    # Parse the ID column: format is "ID_[SOPInstanceUID]_[hemorrhage_type]"
    df['sop_instance_uid'] = df['ID'].apply(lambda x: x.split('_')[1])
    df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])
    
    # Pivot to get one row per image
    pivot_df = df.pivot(index='sop_instance_uid', 
                        columns='hemorrhage_type', 
                        values='Label').reset_index()
    
    return pivot_df


def get_hemorrhage_types(row: pd.Series) -> list:
    """Get list of hemorrhage types present for a given image."""
    hemorrhage_cols = ['epidural', 'subdural', 'subarachnoid', 
                       'intraventricular', 'intraparenchymal']
    
    present = []
    for col in hemorrhage_cols:
        if col in row and row[col] == 1:
            present.append(col)
    
    return present


def process_dataset(input_dir: str,
                    output_dir: str,
                    csv_path: str,
                    num_samples: Optional[int] = None,
                    image_size: int = 512,
                    use_three_channel: bool = True,
                    balance_classes: bool = True,
                    seed: int = 42) -> None:
    """
    Process RSNA dataset for diffusion model training.
    
    Args:
        input_dir: Path to stage_2_train_images folder
        output_dir: Where to save processed images and captions
        csv_path: Path to stage_2_train.csv
        num_samples: Number of samples to process (None for all)
        image_size: Output image size (square)
        use_three_channel: Use 3-channel windowing vs single channel
        balance_classes: Balance positive/negative samples
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse labels
    print("Parsing labels...")
    labels_df = parse_rsna_labels(csv_path)
    
    # Get list of available DICOM files
    dicom_dir = Path(input_dir)
    available_files = {f.stem.replace('ID_', ''): f 
                       for f in dicom_dir.glob('*.dcm')}
    
    # Filter to available files
    labels_df = labels_df[labels_df['sop_instance_uid'].isin(available_files.keys())]
    
    print(f"Found {len(labels_df)} images with labels")
    
    # Balance classes if requested
    if balance_classes and num_samples:
        # Split into positive (any hemorrhage) and negative
        labels_df['has_hemorrhage'] = labels_df['any'] == 1
        
        pos_df = labels_df[labels_df['has_hemorrhage']]
        neg_df = labels_df[~labels_df['has_hemorrhage']]
        
        n_each = num_samples // 2
        
        pos_sample = pos_df.sample(n=min(n_each, len(pos_df)), random_state=seed)
        neg_sample = neg_df.sample(n=min(n_each, len(neg_df)), random_state=seed)
        
        labels_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)
        print(f"Balanced dataset: {len(pos_sample)} positive, {len(neg_sample)} negative")
    
    elif num_samples:
        labels_df = labels_df.sample(n=min(num_samples, len(labels_df)), random_state=seed)
    
    # Process images
    print(f"Processing {len(labels_df)} images...")
    
    metadata_records = []
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        sop_uid = row['sop_instance_uid']
        dicom_path = available_files[sop_uid]
        
        try:
            # Load DICOM
            pixel_array, intercept, slope = load_dicom(str(dicom_path))
            
            # Apply windowing
            if use_three_channel:
                processed = create_three_channel_ct(pixel_array, intercept, slope)
            else:
                processed = get_windowed_image(pixel_array, intercept, slope)
                processed = np.stack([processed] * 3, axis=-1)  # Convert to RGB
            
            # Resize
            img = Image.fromarray(processed)
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            
            # Generate caption
            hemorrhage_types = get_hemorrhage_types(row)
            caption = generate_caption(hemorrhage_types)
            
            # Save image
            output_filename = f"{sop_uid}.png"
            img.save(output_path / output_filename)
            
            # Save caption
            caption_filename = f"{sop_uid}.txt"
            with open(output_path / caption_filename, 'w') as f:
                f.write(caption)
            
            # Record metadata
            metadata_records.append({
                'file_name': output_filename,
                'text': caption,
                'hemorrhage_types': ','.join(hemorrhage_types) if hemorrhage_types else 'none',
                'has_hemorrhage': row['any'] == 1
            })
            
        except Exception as e:
            print(f"Error processing {sop_uid}: {e}")
            continue
    
    # Save metadata JSON for HuggingFace datasets
    import json
    metadata_path = output_path / 'metadata.jsonl'
    with open(metadata_path, 'w') as f:
        for record in metadata_records:
            # Convert numpy types to Python types
            record['has_hemorrhage'] = bool(record['has_hemorrhage'])
            f.write(json.dumps(record) + '\n')
    
    print(f"\nDone! Processed {len(metadata_records)} images")
    print(f"Output saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess RSNA CT dataset for diffusion model fine-tuning'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to stage_2_train_images folder')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to stage_2_train.csv')
    parser.add_argument('--output_dir', type=str, default='./ct_dataset',
                        help='Output directory for processed images')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Output image size (default: 512)')
    parser.add_argument('--single_channel', action='store_true',
                        help='Use single-channel brain window instead of 3-channel')
    parser.add_argument('--no_balance', action='store_true',
                        help='Disable class balancing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        num_samples=args.num_samples,
        image_size=args.image_size,
        use_three_channel=not args.single_channel,
        balance_classes=not args.no_balance,
        seed=args.seed
    )


if __name__ == '__main__':
    main()