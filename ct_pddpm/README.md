# Patched Diffusion Models (pDDPM) for CT Hemorrhage Detection

Implementation of unsupervised anomaly detection in brain CT scans using Patched Diffusion Models.

Based on: [Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI](https://arxiv.org/abs/2303.03758) (Behrendt et al., MIDL 2023)

## Method Overview

The pDDPM approach:
1. **Train on healthy data only**: A diffusion model learns to generate/reconstruct healthy brain CT anatomy
2. **Patched inference**: At test time, divide image into patches, add noise to each patch while keeping surrounding context clean
3. **Context-guided reconstruction**: The model uses clean context to reconstruct each noised patch as "healthy"
4. **Anomaly detection**: Difference between original and reconstruction reveals anomalies (hemorrhages)

Key parameters from the paper:
- Patch size: **60×60 pixels** (optimal)
- Test noise level: **t=400** (out of 1000 timesteps)

## Dataset

RSNA Intracranial Hemorrhage Detection dataset at `/media/M2SSD/gen_models_data/`:
- **752,803 total CT slices**
- **644,870 healthy** (no hemorrhage) - used for training
- **107,933 with hemorrhage** - used for testing anomaly detection
- Hemorrhage types: subdural, intraparenchymal, subarachnoid, intraventricular, epidural

## Files

- `train_pddpm.py` - Custom PyTorch implementation with full control
- `train_pddpm_diffusers.py` - Simplified version using HuggingFace Diffusers
- `inference_pddpm.py` - Anomaly detection inference with evaluation metrics

## Usage

### Training (Diffusers version - recommended for getting started)

```bash
# Quick test with small subset
python train_pddpm_diffusers.py --num_samples 1000 --epochs 20 --batch_size 8

# Full training
python train_pddpm_diffusers.py --num_samples 50000 --epochs 100 --batch_size 16
```

### Training (Custom PyTorch version)

```bash
python train_pddpm.py --num_samples 10000 --epochs 50 --batch_size 16
```

### Anomaly Detection

```bash
# Run detection on test set
python train_pddpm_diffusers.py --mode detect --checkpoint ./checkpoints_diffusers/best_model.pt

# Or with custom implementation
python inference_pddpm.py --checkpoint ./checkpoints/best_model.pt
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `/media/M2SSD/gen_models_data` | Path to RSNA dataset |
| `--image_size` | 256 | Image size (square) |
| `--batch_size` | 16 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--timesteps` | 1000 | Diffusion timesteps |
| `--noise_level` | 400 | Test-time noise level (pDDPM) |
| `--patch_size` | 60 | Patch size for pDDPM inference |

## Results

The anomaly detection outputs:
- **ROC-AUC**: Area under ROC curve (higher is better)
- **PR-AUC**: Area under Precision-Recall curve
- **Anomaly maps**: Pixel-wise visualization of detected anomalies
- **Score distributions**: Comparison of healthy vs hemorrhage scores

## Requirements

```
torch
torchvision
diffusers
pydicom
pandas
numpy
pillow
matplotlib
tqdm
scikit-learn
```

## References

- [arXiv:2303.03758](https://arxiv.org/abs/2303.03758) - Patched Diffusion Models for UAD
- [GitHub: patched-Diffusion-Models-UAD](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD) - Official implementation
