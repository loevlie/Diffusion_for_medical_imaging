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

- `train_from_scratch.py` - trians from scratch
- `train_from_pretrained.py` - Initializes weights using HuggingFace Diffusers
- `evaluate_patched_onestep.py` - Anomaly detection inference with evaluation metrics

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
