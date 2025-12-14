# CT Hemorrhage Detection with Patched Diffusion Models (pDDPM)

Unsupervised anomaly detection for CT brain hemorrhage using patched diffusion models.

Based on: [Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI](https://arxiv.org/abs/2303.03758) (Behrendt et al., MIDL 2023)

![Comparison Plot](comparison.png)

## Method Overview

pDDPM trains a diffusion model on **healthy CT scans only**. At test time:
1. Add noise to image patches (surrounding context stays clean)
2. Denoise using the trained model
3. Compute reconstruction error as anomaly score
4. Hemorrhages produce high reconstruction error (model hasn't learned to reconstruct them)

**Key insight**: Training on healthy data only means the model learns the distribution of normal anatomy. Abnormalities (hemorrhages) fall outside this learned distribution and produce higher reconstruction errors.

## Results Summary

| Model | Best ROC-AUC | Parameters | Input Channels |
|-------|--------------|------------|----------------|
| **From Scratch (1ch)** | **0.7308** | ~14M | 1 (brain window) |
| Pretrained HF (3ch) | 0.6884 | 113.7M | 3 (brain/subdural/bone) |

**Key Finding**: Training from scratch outperforms fine-tuning pretrained models by **+6.2%** ROC-AUC, despite having 8x fewer parameters.

## Architecture Details

### From-Scratch UNet

Custom UNet architecture optimized for single-channel CT images:

```
Input: (B, 1, 256, 256) - Brain window CT scan normalized to [-1, 1]

Encoder:
  - Initial Conv: 1 -> 64 channels
  - Level 1: 64 -> 64 (2 ResBlocks + Downsample)
  - Level 2: 64 -> 128 (2 ResBlocks + Downsample)
  - Level 3: 128 -> 256 (2 ResBlocks + Downsample)
  - Level 4: 256 -> 512 (2 ResBlocks, no downsample)

Middle:
  - ResBlock (512 -> 512)
  - Self-Attention (4 heads)
  - ResBlock (512 -> 512)

Decoder:
  - Level 4: 512+512 -> 256 (2 ResBlocks + Upsample)
  - Level 3: 256+256 -> 128 (2 ResBlocks + Upsample)
  - Level 2: 128+128 -> 64 (2 ResBlocks + Upsample)
  - Level 1: 64+64 -> 64 (2 ResBlocks)
  - Final: GroupNorm + SiLU + Conv(64 -> 1)

Time Embedding: Sinusoidal (64-dim) -> MLP -> 256-dim
Total Parameters: ~14M
```

**ResBlock**: GroupNorm -> SiLU -> Conv -> Time embedding addition -> GroupNorm -> SiLU -> Dropout -> Conv + Skip connection

### Pretrained HuggingFace Model

Uses `google/ddpm-ema-celebahq-256`:
- **Architecture**: UNet2DModel from HuggingFace diffusers
- **Parameters**: 113.7M
- **Input**: 3 channels (adapted for brain/subdural/bone windows)
- **Pre-training**: CelebA-HQ 256x256 face images (EMA weights)

## Training Configuration

### Optimizer & Scheduler
- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate**: 1e-4 (from scratch), 1e-5 (pretrained fine-tuning)
- **LR Scheduler**: Cosine Annealing (T_max = epochs × batches_per_epoch)
- **Gradient Clipping**: Max norm = 1.0

### Diffusion Settings
- **Timesteps**: 1000
- **Noise Schedule**: Cosine (s=0.008)
- **Training Objective**: Predict x0 (clean image) directly

### Patched Training Strategy
During training:
1. Select random 64×64 patch location per sample
2. Add noise only to that patch (rest of image stays clean)
3. Model sees full image but loss computed only on noised patch
4. Forces model to use surrounding context for reconstruction

### Loss Function
Combined MSE + SSIM loss on the noised patches:
```
loss = MSE(pred_patch, orig_patch) + SSIM_loss(pred_patch, orig_patch)
```

### Data Preprocessing
**CT Windowing** (Hounsfield Units to normalized image):
- Brain window: W=80, L=40 (best for soft tissue/hemorrhage)
- Subdural window: W=200, L=80 (for pretrained 3ch model)
- Bone window: W=2800, L=600 (for pretrained 3ch model)

All windows normalized to [0,1] then scaled to [-1,1].

## Requirements

```bash
pip install torch torchvision numpy pandas pydicom pillow scikit-learn matplotlib tqdm diffusers
```

## Data Setup

Uses the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) dataset.

Set `--data_dir` to point to the directory containing:
- `stage_2_train/` - DICOM files (ID_*.dcm)
- `stage_2_train.csv` - Labels

## Quick Start: Replicate Experiments

### 1. Train From Scratch (Recommended)

```bash
python train.py \
    --output_dir checkpoints_scratch \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-4 \
    --num_samples 2000
```

Expected: ROC-AUC ~0.73 by epoch 11-15

### 2. Train From Pretrained (HuggingFace)

```bash
python train.py \
    --pretrained \
    --output_dir checkpoints_pretrained \
    --epochs 15 \
    --batch_size 8 \
    --lr 1e-4
```

Expected: ROC-AUC ~0.66-0.69

### 3. Generate Comparison Plot

```bash
python compare.py \
    --scratch_dir checkpoints_scratch \
    --pretrained_dir checkpoints_pretrained \
    --output comparison.png
```

## Evaluation

### Anomaly Detection Pipeline

At test time (implemented in `evaluate_anomaly_detection()`):

1. For each test image, slide a 64×64 window (stride=32) across the image
2. For each patch:
   - Add noise at level t=150
   - One-step denoise: model predicts x0 directly
   - Compute |original_patch - predicted_patch|
3. Average all patch errors for final anomaly score
4. Higher score = more likely to be anomalous (hemorrhage)

### Run Standalone Evaluation

```bash
python evaluate.py --checkpoint checkpoints_scratch/best_model.pt
```

## Command Line Arguments

### train.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `/media/M2SSD/gen_models_data` | Path to RSNA dataset |
| `--output_dir` | `./checkpoints` | Output directory for checkpoints |
| `--num_samples` | 2000 | Number of healthy training samples |
| `--epochs` | 12 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--eval_samples` | 50 | Samples per class for ROC-AUC eval |
| `--pretrained` | False | Use HuggingFace pretrained model |
| `--hf_model` | `google/ddpm-ema-celebahq-256` | HuggingFace model ID |

### compare.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--scratch_dir` | `checkpoints_patched` | From-scratch checkpoint dir |
| `--pretrained_dir` | `checkpoints_pretrained_3ch` | Pretrained checkpoint dir |
| `--output` | `comparison.png` | Output plot filename |

## Output Files

After training, each checkpoint directory contains:
- `best_model.pt` - Best model by ROC-AUC
- `best_model_loss.pt` - Best model by validation loss
- `training_history.json` - All metrics per epoch
- `training_curves.png` - Loss and ROC-AUC plots
- `diagnostics_epoch_N.png` - Reconstruction visualizations
- `eval_scores_epoch_N.png` - Score distributions and ROC curves

## Why From-Scratch Outperforms Pretrained

1. **Domain Mismatch**: CelebA-HQ faces are vastly different from CT brain scans. The pretrained model has learned features (skin texture, hair, facial features) that don't transfer.

2. **Channel Mismatch**: Pretrained model expects RGB; we adapt by using 3 CT windows, but this is suboptimal compared to a single brain-optimized window.

3. **Overfitting Risk**: The 113M parameter pretrained model may overfit to reconstructing everything well (including anomalies), reducing detection capability.

4. **Task-Specific Features**: Training from scratch on CT data learns anatomy-specific features directly useful for anomaly detection.

## File Structure

```
ct_pddpm/
├── train.py           # Main training script (scratch or pretrained)
├── evaluate.py        # Standalone evaluation
├── compare.py         # Generate comparison plots
├── README.md          # This file
├── checkpoints_*/     # Output directories with models and metrics
└── comparison.png     # Generated comparison plot
```

## References

- [arXiv:2303.03758](https://arxiv.org/abs/2303.03758) - Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI
- [GitHub: patched-Diffusion-Models-UAD](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD) - Official implementation
- [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) - Dataset
- [HuggingFace diffusers](https://huggingface.co/docs/diffusers/) - Pretrained diffusion models

## Citation

If using this code, please cite the original pDDPM paper:

```bibtex
@inproceedings{behrendt2023patched,
  title={Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI},
  author={Behrendt, Finn and Bhattacharya, Debayan and Kr{\"u}ger, Julia and Opfer, Roland and Schlaefer, Alexander},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
```
