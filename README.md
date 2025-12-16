# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **intrusion detection system** for the UNSW-NB15 network traffic dataset using a deep learning autoencoder approach. The project trains an anomaly detection model to identify network attacks by learning to reconstruct normal traffic patterns.

**Key Architecture**: Hybrid CNN + Bidirectional LSTM Autoencoder
- Trains only on normal traffic (semi-supervised)
- Detects attacks via reconstruction error
- Uses sequence-based analysis for temporal patterns

## Running the Project

### Environment Setup

This project requires Jupyter notebook support. All code is in a single notebook file.

```bash
# Install dependencies (no requirements.txt exists yet)
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

# Run the notebook in Jupyter
jupyter notebook eeg-identification.ipynb
```

**Note**: Despite the filename, this notebook contains network intrusion detection code, not EEG analysis. The naming is legacy from a previous project iteration.

### Working with the Notebook

The notebook [eeg-identification.ipynb](eeg-identification.ipynb) contains the complete pipeline:

1. **Cell 0**: Import dependencies and set random seeds
2. **Cell 1**: Data preprocessing functions (categorical encoding, log transforms)
3. **Cell 2**: Load UNSW-NB15 dataset, apply scaling and feature selection
4. **Cell 3**: Create time sequences from network traffic data
5. **Cell 4**: Build and train autoencoder model with residual blocks
6. **Cell 5**: Evaluate model using hybrid error metric (MSE + cosine distance)
7. **Cell 6**: Generate visualizations (training loss, error distributions)

### Dataset Requirements

The notebook expects UNSW-NB15 dataset files at:
- `/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv`
- `/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv`

**When modifying**: Update the paths in cell 2 if running outside Kaggle environment.

## Architecture Details

### Preprocessing Pipeline

**Categorical Features**: Top-K encoding for `proto`, `service`, `state` columns (keeps top 6 values, rest become "other")

**Numeric Features**: Log transformation for skewed features:
- Duration, bytes (source/dest), load (source/dest)
- Packets (source/dest), TTL (source/dest)

**Scaling**: RobustScaler with quantile range (5, 95) - robust to outliers in network traffic

**Feature Selection**: VarianceThreshold removes near-constant features (< 1e-4 variance)

**Final Feature Count**: ~57 features after encoding and selection

### Sequence Creation

**Purpose**: Convert packet-level features into temporal sequences for RNN processing

**Parameters**:
- `SEQ_LEN = 16`: Each sequence contains 16 consecutive packets
- `STRIDE = 2`: Sequences overlap (sliding window)
- **Labeling strategy**: Sequence labeled as attack if >30% of packets are attacks

**Training**: Uses only normal traffic sequences (unsupervised anomaly detection)

### Model Architecture

**Encoder Path**:
```
Input (seq_len=16, features=57)
  ↓
Conv1D(64, 3) + Residual Block + MaxPool
  ↓
Residual Block(128) + MaxPool
  ↓
Bidirectional LSTM(128)
  ↓
Dense(96, latent representation)
```

**Decoder Path**:
```
Latent(96)
  ↓
RepeatVector + Bidirectional LSTM(128)
  ↓
UpSampling + Residual Block(128)
  ↓
UpSampling + Residual Block(64)
  ↓
TimeDistributed Dense(57, reconstructed output)
```

**Residual Blocks**: Each contains:
- Conv1D(filters, 3) + BatchNorm + Dropout(0.2)
- Conv1D(filters, 3) + BatchNorm
- Skip connection with 1x1 Conv if needed
- ReLU activation

### Training Strategy

**Objective**: Minimize MSE between input and reconstructed output

**Denoising**: Adds 2% Gaussian noise to inputs during training (improves robustness)

**Callbacks**:
- EarlyStopping(patience=6): Stops if validation loss doesn't improve
- ReduceLROnPlateau(patience=4, factor=0.5): Reduces learning rate when stuck

**Hyperparameters**:
- Optimizer: Adam(5e-4)
- Batch size: 64
- Max epochs: 40
- Validation split: 20% of normal traffic

### Evaluation Method

**Hybrid Error Metric**: Combines two distance measures
- **MSE (70%)**: Mean squared error across sequence and features
- **Cosine distance (30%)**: 1 - cosine similarity between flattened vectors

**Threshold Selection**: Uses precision-recall curve to find threshold maximizing F1-score

**Attack Detection**: Traffic with reconstruction error > threshold is classified as attack

**Expected Performance**: ~91% accuracy, ~0.93 ROC-AUC on UNSW-NB15 test set

## Common Development Tasks

### Modifying the Model Architecture

The model is defined in cell 4 with two main functions:
- `residual_block(x, filters, dropout=0.2)` - Define residual blocks
- `build_model(seq_len, n_features, latent_dim=96)` - Full autoencoder

**To change model capacity**: Adjust filter sizes, latent_dim, or add/remove blocks

**To modify sequence length**: Change `SEQ_LEN` in cell 3 and retrain

### Adjusting Preprocessing

Preprocessing logic is in cell 1 functions:
- `handle_categorical_top_k(df, col, k=6)` - Modify k to keep more/fewer categories
- `load_and_preprocess(train_path, test_path)` - Add new preprocessing steps here

Feature engineering in cell 2:
- Add new features before `scaler.fit_transform()`
- Adjust variance threshold if keeping too few/many features

### Hyperparameter Tuning

Key parameters to experiment with:
- **Sequence length** (SEQ_LEN): Longer captures more temporal context
- **Stride**: Smaller creates more overlapping sequences
- **Latent dimension**: Bottleneck size (96 default)
- **Learning rate**: Currently 5e-4
- **Batch size**: 64 (larger may stabilize training)
- **Noise level**: 0.02 in `add_noise()` function
- **Error metric weights**: 0.7 MSE + 0.3 cosine in cell 5

### Running Experiments

Since this is a single notebook:
1. Make modifications in relevant cells
2. Run from top to bottom (Kernel → Restart & Run All)
3. Training takes ~10-15 minutes on GPU, longer on CPU
4. Check final metrics in cell 5 output

**Tip**: For quick iterations, train on a subset by adding filtering in cell 3:
```python
# Example: Use only first 10000 samples for fast testing
X_train = X_train[:10000]
y_train = y_train[:10000]
```

## Important Technical Notes

### Data Imbalance

The UNSW-NB15 dataset is heavily imbalanced (more attacks than normal traffic in test set). The autoencoder approach handles this by training only on normal traffic, making it suitable for this scenario.

### Sequence Labeling Strategy

The 30% threshold in `create_sequences()` is heuristic. Adjusting this affects how mixed sequences are labeled:
- Lower threshold (e.g., 10%): More sequences labeled as attacks
- Higher threshold (e.g., 50%): Requires majority attack packets

### Why Hybrid Error Metric?

- **MSE alone**: Good for magnitude differences, sensitive to outliers
- **Cosine distance alone**: Good for direction differences, ignores magnitude
- **Combined**: Captures both aspects of reconstruction quality

The 0.7/0.3 weighting emphasizes magnitude (MSE) while still considering directionality.

### Model Capacity

With ~500K-1M parameters, this model is moderately sized:
- **Sufficient** for the 57-feature UNSW-NB15 dataset
- **Avoid**: Drastically increasing depth without more data (risk overfitting)
- **Consider**: Reducing if training time is concern

## Git Repository Context

### Current State

The repository has transitioned from a multi-notebook structure to a single consolidated notebook:

- **Previous**: Separate notebooks for preprocessing, training, evaluation (now deleted)
- **Previous**: Extensive documentation (README, PROJECT_SUMMARY, QUICK_START) (now deleted)
- **Current**: Single notebook `eeg-identification.ipynb` with complete pipeline
- **Current**: `logs/` and `models/` directories exist but are empty

### Important History

The git status shows many deleted files (previous project iteration). The commit history shows:
1. Initial setup (845e06a)
2. Data preprocessing (11d69ff)
3. Added model/log folders (7189c85)
4. Consolidated to single notebook (e9cb517)

**When referencing old documentation**: It's been deleted but exists in git history. Use:
```bash
git show 845e06a:README.md         # View original README
git show e9cb517:PROJECT_SUMMARY.md  # View project summary
```

## Working in This Codebase

### File Naming Caveat

The notebook is named `eeg-identification.ipynb` but actually contains network intrusion detection code using UNSW-NB15 dataset. This is a naming artifact from repository history.

**Do not be confused**: References to "EEG" in filenames are legacy. The actual work is network security/intrusion detection.

### Making Changes

Since everything is in one notebook:
1. Cell dependencies matter - run cells in order
2. Modifying early cells (preprocessing) requires re-running all downstream cells
3. Model training (cell 4) takes the longest - save results if experimenting
4. Use Jupyter's "Run All" feature for clean end-to-end execution

### Code Organization

The notebook structure follows ML pipeline conventions:
- Imports → Preprocessing → Data Loading → Sequence Creation → Model Definition → Training → Evaluation

Each cell is relatively self-contained but depends on variables from previous cells.

### TensorFlow/Keras Usage

The code uses TensorFlow 2.x with Keras API:
- Functional API for model building (not Sequential)
- Callbacks for training control
- Uses `tf.random.set_seed(42)` for reproducibility

**GPU**: Automatically uses GPU if available. Check with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Future Development Considerations

### Missing Components

The repository currently lacks:
- Requirements file (dependencies not specified)
- Model persistence (training doesn't save models)
- Data loading flexibility (hardcoded Kaggle paths)
- Comprehensive documentation (previous docs deleted)

**Recommendations for improvements**:
1. Add requirements.txt with pinned versions
2. Add model saving logic in cell 4 after training
3. Make dataset paths configurable (environment variables or config)
4. Consider splitting notebook for modularity if it grows larger

### Potential Extensions

Based on the architecture:
- **Multi-class attack classification**: Classify attack types instead of binary detection
- **Real-time inference**: Optimize for streaming network traffic
- **Feature importance**: Analyze which network features most indicate attacks
- **Ensemble methods**: Train multiple autoencoders with different architectures
- **Transfer learning**: Pre-train on related network security datasets

### Performance Optimization

If training is slow:
- Reduce sequence length (SEQ_LEN)
- Use mixed precision training (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`)
- Batch size adjustment based on available memory
- Profile to identify bottlenecks (data loading vs computation)

## Dataset Information

**UNSW-NB15**: Modern network intrusion detection dataset created by UNSW Canberra
- Captures real modern network traffic
- Contains 9 attack categories
- More realistic than older datasets (KDD Cup, NSL-KDD)
- Features include flow statistics, packet headers, content-based features

**Citation**: Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems.

## Troubleshooting

### Common Issues

**"Cannot find UNSW-NB15 files"**: Update paths in cell 2 to point to your local dataset

**Out of memory during training**: Reduce batch size in cell 4 or sequence length in cell 3

**Poor performance**: Check that scaling is applied correctly and feature selection isn't removing too many features

**Model not converging**: Adjust learning rate, increase epochs, or check for data preprocessing issues

**GPU not being used**: Verify TensorFlow-GPU installation and CUDA compatibility

## Summary

This is a **semi-supervised anomaly detection system** for network intrusion detection using a sequence-based autoencoder. The entire pipeline exists in a single Jupyter notebook that trains on normal traffic and detects attacks via reconstruction error. The architecture combines CNN spatial feature extraction with LSTM temporal modeling, achieving strong performance on the UNSW-NB15 dataset.
