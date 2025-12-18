# Brain Tumor Segmentation Project

This project implements brain tumor segmentation using MRI images with two different approaches:
1. **Custom Model**: A purpose-built U-Net architecture
2. **Pre-trained Model**: Transfer learning using ResNet or VGG encoders

## Project Structure

```
image_processing/
├── data/                      # Data directories (created automatically)
│   ├── raw/                  # Original datasets
│   ├── processed/            # Preprocessed data
│   ├── external/             # Additional datasets
│   └── splits/               # Train/val/test splits
├── models/                   # Model architectures
│   ├── custom_model.py      # Custom U-Net
│   ├── pretrained_model.py  # Pre-trained models
│   └── saved/               # Trained model checkpoints
├── preprocessing/           # Image preprocessing
│   ├── denoising.py         # Denoising methods
│   ├── normalization.py    # Normalization methods
│   └── augmentation.py      # Data augmentation
├── training/                # Training utilities
│   ├── trainer.py          # Training loop
│   └── losses.py           # Loss functions
├── evaluation/             # Evaluation metrics
│   └── metrics.py         # DSC, IoU, etc.
├── utils/                  # Utilities
│   ├── config.py         # Configuration
│   └── data_loader.py    # HDF5 data loading
├── train_custom.py        # Train custom model
├── train_pretrained.py   # Train pre-trained model
├── evaluate_models.py    # Evaluate and compare models
└── requirements.txt      # Python dependencies
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify PyTorch installation**
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Dataset

The project expects an HDF5 dataset at:
```
dataset/brain_tumor_dataset.h5
```

The HDF5 file should contain patient groups with:
- `image`: 2D MRI brain slice (512×512 or 256×256 grayscale)
- `tumor_mask`: Binary segmentation mask
- `label`: Tumor type (1=Meningioma, 2=Glioma, 3=Pituitary)

## Usage

### 1. Train Custom Model

```bash
python train_custom.py
```

This will:
- Explore the dataset
- Create train/validation/test splits
- Apply preprocessing and augmentation
- Train the custom U-Net model
- Save checkpoints and logs

### 2. Train Pre-trained Model

```bash
python train_pretrained.py
```

This will:
- Use the same data splits
- Load a pre-trained ResNet or VGG encoder
- Fine-tune for brain tumor segmentation
- Save checkpoints and logs

### 3. Evaluate Models

```bash
python evaluate_models.py
```

This will:
- Load both trained models
- Evaluate on test set
- Compute metrics (DSC, IoU, etc.)
- Compare models and save results

## Configuration

Edit `utils/config.py` to adjust:
- Image size (default: 256×256)
- Batch size (default: 16)
- Learning rate (default: 1e-4)
- Denoising method (default: bilateral)
- Normalization method (default: z_score)
- Model architecture parameters
- Training hyperparameters

## Evaluation Metrics

The primary metric is **Dice Similarity Coefficient (DSC)**:
- DSC = 1: Perfect overlap
- DSC = 0: No overlap
- Target: DSC > 0.8

Additional metrics computed:
- IoU (Intersection over Union)
- Pixel Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score

## Model Checkpoints

Trained models are saved in `models/saved/`:
- `custom_unet_best.pth`: Best custom model
- `pretrained_*_best.pth`: Best pre-trained model
- `*_latest.pth`: Latest checkpoint

## TensorBoard Logging

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir logs/
```

View training curves, loss, and metrics in your browser.

## Results

Evaluation results are saved in `results/` as JSON files containing:
- Per-model metrics
- Comparison statistics
- Overall winner determination

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE` (e.g., 256 instead of 512)
- Use gradient accumulation

### Slow Training
- Reduce `NUM_WORKERS` if data loading is slow
- Use mixed precision training (enabled by default)
- Consider using smaller model

### Poor Performance
- Try different denoising methods
- Adjust augmentation parameters
- Increase training epochs
- Try different loss functions (CombinedLoss)

## Citation

If you use this code, please cite the following papers:

### U-Net Architecture
```
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
In Medical Image Computing and Computer-Assisted Intervention (MICCAI) (pp. 234-241). Springer.
```

### ResNet Architecture
```
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770-778).
```

### VGG Architecture
```
Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. 
arXiv preprint arXiv:1409.1556.
```

### Dice Similarity Coefficient
```
Dice, L. R. (1945). Measures of the amount of ecologic association between species. 
Ecology, 26(3), 297-302.
```

### BibTeX Format
```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={234--241},
  year={2015},
  organization={Springer}
}

@inproceedings{he2016resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}

@article{simonyan2014vgg,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}

@article{dice1945measures,
  title={Measures of the amount of ecologic association between species},
  author={Dice, Lee R},
  journal={Ecology},
  volume={26},
  number={3},
  pages={297--302},
  year={1945}
}
```

## License

[Add your license here]

## Contact

[Add contact information]

