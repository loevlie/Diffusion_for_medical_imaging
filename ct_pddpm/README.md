# CT Hemorrhage Detection with Patched Diffusion Models (pDDPM)

Unsupervised anomaly detection for CT brain hemorrhage using patched diffusion models.

Based on: [Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI](https://arxiv.org/abs/2303.03758) (Behrendt et al., MIDL 2023)

## Method Overview

pDDPM trains a diffusion model on **healthy CT scans only**. At test time:
1. Add noise to image patches (surrounding context stays clean)
2. Denoise using the trained model
3. Compute reconstruction error as anomaly score
4. Hemorrhages produce high reconstruction error (model hasn't learned to reconstruct them)

**Key insight**: Training on healthy data only means the model learns the distribution of normal anatomy. Abnormalities (hemorrhages) fall outside this learned distribution and produce higher reconstruction errors.

## Results

| Model | Best ROC-AUC | Parameters |
|-------|--------------|------------|
| Scratch UNet | 0.722 | ~30M |
| Pretrained (HuggingFace) | 0.676 | ~114M |

## Architecture

### UNet (from scratch)
- **Channels**: 64 base, multipliers (1, 2, 4, 8) -> [64, 128, 256, 512]
- **Timesteps**: 1000 with cosine noise schedule
- **Parameters**: ~30M
- **Input**: Single channel (brain window CT)

### Pretrained Model
- **Base**: `google/ddpm-ema-celebahq-256` from HuggingFace
- **Parameters**: ~114M
- **Input**: 3 channels (brain/subdural/bone windows)

## Installation

```bash
pip install torch torchvision numpy pandas pydicom pillow scikit-learn matplotlib tqdm diffusers
```

## Data

Uses the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) dataset.

Required structure:
```
data_dir/
├── stage_2_train/      # DICOM files (ID_*.dcm)
└── stage_2_train.csv   # Labels
```

## Usage

### Train from scratch
```bash
python train_patched.py --output_dir checkpoints --epochs 10 --batch_size 8 --num_samples 2000
```

### Train with pretrained model
```bash
python train_patched.py --pretrained --output_dir checkpoints_pretrained --epochs 10 --batch_size 8
```

### Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## File Structure

```
ct_pddpm/
├── train_patched.py            # Main training script
├── evaluate.py                 # Evaluation script
├── create_presentation_figures.py  # Generate presentation visualizations
├── generate_clean_examples.py  # Generate clean error map examples
├── visualize_segmentation.py   # Visualize segmentation results
├── checkpoints_patched/        # Trained scratch model
├── checkpoints_pretrained/     # Trained pretrained model
└── presentation_figures/       # Output figures
```

## References

- [arXiv:2303.03758](https://arxiv.org/abs/2303.03758) - Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI
- [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) - Dataset

## Citation

```bibtex
@inproceedings{behrendt2023patched,
  title={Patched Diffusion Models for Unsupervised Anomaly Detection in Brain MRI},
  author={Behrendt, Finn and Bhattacharya, Debayan and Kr{\"u}ger, Julia and Opfer, Roland and Schlaefer, Alexander},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
```
