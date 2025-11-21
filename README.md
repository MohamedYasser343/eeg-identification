# EEG Person Identification using CNN + RNN

A deep learning approach to identify individuals based on their unique EEG (electroencephalography) brainwave patterns using a hybrid CNN+RNN architecture.

## Project Overview

This project implements a person identification system that uses time-frequency analysis of EEG signals combined with deep learning to classify 109 different subjects from the PhysioNet Motor Movement/Imagery Dataset.

### Key Features
- **Dataset**: PhysioNet EEG Motor Movement/Imagery Database (109 subjects)
- **Architecture**: Hybrid CNN + Bidirectional LSTM model
- **Input**: Time-frequency spectrograms from 64-channel EEG recordings
- **Task**: Multi-class classification (109 subjects)
- **Performance**: Achieves high accuracy with comprehensive evaluation metrics

## Architecture

```
EEG Signal → Preprocessing → Spectrogram → CNN → RNN → Classification
```

### Model Components
1. **CNN Feature Extractor**: Extracts spatial-temporal-frequency patterns from spectrograms
2. **Bidirectional LSTM**: Captures temporal dependencies across time sequences
3. **Classification Head**: Dense layers with softmax for 109-class output

## Project Structure

```
EEG_Identification/
│
├── 1_preprocessing.ipynb          # Data loading, filtering, and spectrogram generation
├── 2_model_training.ipynb         # CNN+RNN model architecture and training
├── 3_performance_report.ipynb     # Evaluation, visualization, and analysis
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Data directory (created during execution)
│   └── processed/                 # Preprocessed data files
│       ├── preprocessed_data.h5   # Train/val/test splits
│       └── config.pkl             # Configuration parameters
│
├── models/                        # Trained models (created during training)
│   ├── *.keras                    # Saved model weights
│   ├── *_results.json            # Training results
│   └── *_performance_report.csv  # Per-subject metrics
│
├── figures/                       # Visualizations (created during execution)
│   ├── raw_eeg_sample.png
│   ├── sample_spectrograms.png
│   ├── training_history.png
│   ├── confusion_matrix_*.png
│   ├── per_subject_analysis.png
│   ├── tsne_visualization.png
│   └── error_analysis.png
│
└── logs/                          # TensorBoard logs (created during training)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended but not required)
- At least 8GB RAM
- 10GB free disk space for data and models

### Setup

1. **Clone or download this project:**
```bash
cd EEG_Identification
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

Key libraries:
- `mne` - EEG data processing
- `tensorflow` - Deep learning framework
- `numpy`, `scipy`, `pandas` - Numerical computing
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning utilities

See [requirements.txt](requirements.txt) for complete list.

## Usage

### Quick Start

Run the notebooks in sequence:

#### 1. Data Preprocessing
```bash
jupyter notebook 1_preprocessing.ipynb
```

**What it does:**
- Downloads PhysioNet EEG dataset (automatic download via MNE)
- Applies bandpass filtering (0.5-50 Hz)
- Segments continuous EEG into 3-second epochs
- Computes time-frequency spectrograms using STFT
- Splits data into train/validation/test sets (70/15/15)
- Saves preprocessed data to `data/processed/`

**Time estimate:** 30-60 minutes (depending on hardware)

#### 2. Model Training
```bash
jupyter notebook 2_model_training.ipynb
```

**What it does:**
- Loads preprocessed data
- Builds CNN+RNN hybrid architecture
- Trains model with callbacks (early stopping, learning rate scheduling)
- Saves best model to `models/`
- Generates training visualizations

**Time estimate:** 1-3 hours (depending on GPU/CPU)

#### 3. Performance Evaluation
```bash
jupyter notebook 3_performance_report.ipynb
```

**What it does:**
- Evaluates trained model on test set
- Generates confusion matrix
- Per-subject performance analysis
- t-SNE visualization of learned features
- Error analysis and model calibration
- Comprehensive discussion and insights

**Time estimate:** 10-20 minutes

## Results

### Expected Performance

Based on the architecture and dataset:

- **Test Accuracy**: ~70-85% (varies with training)
- **Top-5 Accuracy**: ~90-95%
- **F1-Score (Macro)**: ~0.65-0.80

### Deliverables

1. **Preprocessing Notebook**: Complete pipeline from raw EEG to spectrograms
2. **Training Notebook**: Trained CNN+RNN model with saved weights
3. **Performance Report**: Comprehensive analysis including:
   - Confusion matrix (109×109)
   - Per-subject accuracy and F1-scores
   - Training/validation curves
   - t-SNE feature visualization
   - Error analysis
   - Discussion of results

## Model Details

### Preprocessing Parameters
- **Sampling Rate**: 160 Hz
- **Bandpass Filter**: 0.5-50 Hz
- **Epoch Duration**: 3 seconds
- **Overlap**: 50%
- **STFT Window**: 256 samples
- **Hop Length**: 32 samples
- **Frequency Bins**: 50

### Model Architecture
```python
Input: (64 channels, 50 freq_bins, time_bins, 1)
↓
Conv2D(64, 5×5) → BatchNorm → MaxPool → Dropout(0.3)
↓
Conv2D(128, 3×3) → BatchNorm → MaxPool → Dropout(0.3)
↓
Reshape → Bidirectional LSTM(128) → Dropout(0.4)
↓
Dense(256) → BatchNorm → Dropout(0.5)
↓
Dense(109, softmax)
```

### Training Configuration
- **Optimizer**: Adam (initial lr=0.001)
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Max Epochs**: 100 (with early stopping)
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Dataset Information

### PhysioNet Motor Movement/Imagery Dataset

- **Subjects**: 109 healthy individuals
- **Sessions**: 2 per subject (different days)
- **Channels**: 64 EEG electrodes (10-10 system)
- **Sampling Rate**: 160 Hz
- **Tasks**: Motor imagery (left hand, right hand, both hands, both feet)
- **Runs Used**: 3, 7, 11 (motor imagery tasks)

**Citation:**
```
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
IEEE Transactions on Biomedical Engineering, 51(6):1034-1043, 2004.
```

Dataset is automatically downloaded via MNE-Python when running the preprocessing notebook.

## Visualization Examples

The project generates multiple visualizations:

1. **Raw EEG Signals**: Time-series plots of multichannel EEG
2. **Spectrograms**: Time-frequency representations of brain activity
3. **Training History**: Loss and accuracy curves
4. **Confusion Matrix**: 109×109 heatmap of predictions
5. **Per-Subject Performance**: Bar plots and scatter plots
6. **t-SNE Embeddings**: 2D visualization of learned features
7. **Error Analysis**: Confidence distributions and calibration curves

All figures are saved to `figures/` directory.

## Potential Improvements

### Data Augmentation
- Time warping
- Frequency masking
- Adding noise
- Mixup/CutMix strategies

### Architecture Enhancements
- Attention mechanisms (spatial/temporal/channel)
- Multi-scale feature extraction
- Residual connections
- Transformer-based models

### Training Strategies
- Cross-session validation
- Subject-specific fine-tuning
- Ensemble methods
- Transfer learning from pre-trained models

### Advanced Analysis
- Grad-CAM for interpretability
- Frequency band analysis (alpha, beta, gamma)
- Cross-dataset generalization
- Real-time inference optimization

## Applications

- **Biometric Authentication**: Secure access using brainwave patterns
- **Security Systems**: Brain-based identification for high-security environments
- **Neuroscience Research**: Study individual differences in brain activity
- **Clinical Applications**: Patient identification in medical settings
- **BCI Systems**: Personalized brain-computer interfaces

## Troubleshooting

### Common Issues

1. **Memory Error during preprocessing:**
   - Reduce `n_subjects_to_use` in config (try 50 instead of 109)
   - Process subjects in batches

2. **GPU Out of Memory:**
   - Reduce batch size in training
   - Use gradient accumulation
   - Enable mixed precision training

3. **Slow training:**
   - Ensure GPU is being used (check with `tf.config.list_physical_devices('GPU')`)
   - Reduce model complexity
   - Use smaller subset for initial testing

4. **Poor performance:**
   - Increase training epochs
   - Adjust learning rate
   - Try different preprocessing parameters
   - Ensure data is properly normalized

## Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| GPU | None (CPU mode) | NVIDIA GPU with 6GB+ VRAM |
| Storage | 10GB | 20GB+ |

### Timing Estimates

| Stage | CPU | GPU (NVIDIA RTX 3070) |
|-------|-----|------------------------|
| Preprocessing | 45-60 min | 30-45 min |
| Training (100 epochs) | 4-6 hours | 1-2 hours |
| Evaluation | 15-20 min | 5-10 min |

## Contributing

Suggestions for improvements:
- Implement additional data augmentation techniques
- Add cross-session evaluation
- Integrate attention visualization
- Develop real-time inference pipeline
- Add support for other EEG datasets

## License

This project is for educational purposes. The PhysioNet dataset has its own usage terms.

## References

1. Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface System.
2. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
3. Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python.

## Contact

For questions or issues, please open an issue in the repository.

---

**Project Status**: Complete and ready for use

**Last Updated**: 2025

**Version**: 1.0
