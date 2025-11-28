# U-Net Medical Image Segmentation

A deep learning project for medical image segmentation using U-Net architecture with TensorFlow/Keras. Designed to segment lesions or regions of interest from 3D volumetric MRI data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Metrics](#metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a U-Net convolutional neural network for medical image segmentation. It's specifically designed to:
- Load and preprocess 3D volumetric MRI data (NIfTI format)
- Extract 2D slices from volumetric data
- Apply data augmentation (flipping, rotation)
- Train a U-Net model for binary segmentation
- Evaluate model performance using Dice coefficient and IoU metrics
- Visualize predictions on validation data

**Use Case**: Lesion/tumor segmentation from brain MRI scans

---

## ✨ Features

- **Data Loading**: Support for NIfTI volumetric data via `nibabel`
- **Preprocessing**: Normalization, resizing, and slice extraction
- **Augmentation**: Horizontal/vertical flips and random rotations
- **U-Net Architecture**: Encoder-decoder with skip connections
- **Custom Metrics**: Dice coefficient and Intersection over Union (IoU)
- **Training Features**:
  - Early stopping to prevent overfitting
  - Model checkpointing to save best weights
  - Learning rate scheduling (ReduceLROnPlateau)
- **Evaluation**: Model testing with visualization of predictions
- **Command-line Interface**: Pass training parameters via arguments

---

## 📁 Project Structure

```
Segmenter/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                    # Raw MRI files (NIfTI format)
│   ├── preprocess/             # Preprocessed data (numpy arrays)
│   └── masks/                  # Ground truth mask files
├── outputs/
│   └── best_model_params/      # Trained model and checkpoints
├── src/
│   ├── config.py              # Configuration settings
│   ├── preprocess.py          # Data preprocessing functions
│   ├── dataset.py             # Dataset loading and augmentation
│   ├── model.py               # U-Net model architecture
│   ├── metrics.py             # Custom metrics (Dice, IoU)
│   ├── train.py               # Training script
│   └── evaluation.py          # Model evaluation script
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- macOS, Linux, or Windows with pip

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/u-net-segmentation.git
   cd u-net-segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Dependencies
- TensorFlow >= 2.15.0
- Keras >= 2.15.0
- NumPy >= 1.23.5
- nibabel >= 5.0.0 (for NIfTI file handling)
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- TensorFlow Addons >= 0.23.0

---

## ⚙️ Configuration

Edit `src/config.py` to adjust training parameters:

```python
class Config:
    # Training parameters
    num_epochs = 10              # Number of training epochs
    learning_rate = 1e-4         # Adam optimizer learning rate
    batch_size = 20              # Batch size for training
    validation_split = 0.2       # 80/20 train-validation split
    base_filters = 64            # Base filters in first Conv layer
    
    # Data augmentation
    augmentation = {
        "flip_horizontal": True,
        "flip_vertical": True,
        "rotation_angle": 10,    # Random rotation ±10 degrees
        "zoom": 0.1
    }
    
    # Image parameters
    IMG_height = 128
    IMG_width = 128
    IMG_channel = 1              # Grayscale MRI
    
    # Paths
    Data_dir = "data"
    Processed_dir = "data/preprocess"
    Output_dir = "outputs"
    model_dir = "best_model_params"
```

---

## 📊 Usage

### 1. Prepare Data

Place your raw MRI files in `data/raw/` directory:
```
data/
├── raw/
│   ├── patient_001.nii.gz
│   ├── patient_002.nii.gz
│   └── ...
└── masks/
    ├── patient_001_mask.nii.gz
    ├── patient_002_mask.nii.gz
    └── ...
```

### 2. Preprocess Data

Process volumetric data into 2D slices:
```bash
cd src
python preprocess.py
```

This will:
- Load NIfTI files
- Normalize intensities
- Extract 2D slices
- Save as NumPy arrays in `data/preprocess/`

---

## 🎓 Training

### Basic Training

```bash
cd src
python train.py
```

Uses default config values from `config.py`.

### Training with Custom Parameters

Override config settings via command-line arguments:

```bash
python train.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --model_path /path/to/models \
    --data_path /path/to/data
```

### Command-line Arguments
- `--epochs` (int): Number of training epochs
- `--batch_size` (int): Batch size for training
- `--learning_rate` (float): Adam optimizer learning rate
- `--model_path` (str): Directory to save model checkpoints
- `--data_path` (str): Path to preprocessed training data

### Training Output

Training will display:
- Model architecture summary
- Training progress with loss and metrics
- Validation performance
- Saved checkpoints to `outputs/best_model_params/`

The script saves:
- `best_model_lesion` - Model with best validation Dice score
- `final_model.keras` - Final model after all epochs

---

## 📈 Evaluation

### Evaluate Trained Model

```bash
cd src
python evaluation.py
```

This will:
1. Load the best trained model
2. Evaluate on validation dataset
3. Report metrics (Dice, IoU, Precision, Recall)
4. Visualize predictions with overlays

### Output Example
```
name: loss, score: 0.2341
name: dice_coeff, score: 0.8234
name: iou_metric, score: 0.7892
name: precision, score: 0.8456
name: recall, score: 0.8123
```

---

## 🏗️ Model Architecture

**U-Net with 4 encoder levels and 4 decoder levels:**

```
Input (128x128x1)
    ↓
Encoder Block 1: Conv2D(64) → BatchNorm → ReLU (128×128×64)
    ↓ MaxPool
Encoder Block 2: Conv2D(128) → BatchNorm → ReLU (64×64×128)
    ↓ MaxPool
Encoder Block 3: Conv2D(256) → BatchNorm → ReLU (32×32×256)
    ↓ MaxPool
Encoder Block 4: Conv2D(512) → BatchNorm → ReLU (16×16×512)
    ↓ Conv2DTranspose
Decoder Block 3: Concatenate + Conv2D(256) (32×32×256)
    ↓ Conv2DTranspose
Decoder Block 2: Concatenate + Conv2D(128) (64×64×128)
    ↓ Conv2DTranspose
Decoder Block 1: Concatenate + Conv2D(64) (128×128×64)
    ↓
Output: Conv2D(1, sigmoid) → Binary segmentation (128×128×1)
```

**Skip Connections**: Features from encoder are concatenated with decoder features to preserve spatial information.

---

## 📊 Metrics

### Dice Coefficient
Measures overlap between prediction and ground truth:
```
Dice = (2 * |A ∩ B|) / (|A| + |B|)
Range: [0, 1] where 1 is perfect overlap
```

### Intersection over Union (IoU)
Also known as Jaccard Index:
```
IoU = |A ∩ B| / |A ∪ B|
Range: [0, 1] where 1 is perfect overlap
```

### Precision & Recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source myvenv/bin/activate
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: data/preprocess/train_image.npy`
**Solution**: Run preprocessing first:
```bash
cd src
python preprocess.py
```

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size:
```bash
python train.py --batch_size 8
```

### Issue: Model not improving
**Solution**: Try adjusting learning rate:
```bash
python train.py --learning_rate 1e-5 --epochs 50
```

---

## 📚 Additional Resources

- [U-Net Paper](https://arxiv.org/abs/1505.04597): U-Net: Convolutional Networks for Biomedical Image Segmentation
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras API Reference](https://keras.io/)
- [nibabel Documentation](https://nipy.org/nibabel/)

---

## 👤 Author

Created for medical image segmentation research and development.

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## ❓ FAQ

**Q: Can I use this for 3D segmentation directly?**
A: Currently designed for 2D slices. For 3D volumetric segmentation, modify the model to use Conv3D layers.

**Q: What's the recommended GPU?**
A: Works on CPU, but GPU recommended for faster training. Tested on Apple Silicon (M1/M2).

**Q: Can I use different image sizes?**
A: Yes, modify `IMG_height` and `IMG_width` in `config.py`. Keep dimensions as multiples of 16 for U-Net.

**Q: How do I handle class imbalance?**
A: Currently uses binary crossentropy. For severe imbalance, try weighted loss or focal loss.

---

## ✅ Recent Updates

- ✅ Fixed tensor reshape indexing in metrics
- ✅ Corrected data augmentation logic
- ✅ Fixed encoder prefix consistency in model
- ✅ Unified file naming across preprocess/dataset modules
- ✅ Added proper error handling for data loading
- ✅ Improved type hints and documentation

