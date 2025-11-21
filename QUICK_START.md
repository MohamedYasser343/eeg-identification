# Quick Start Guide - EEG Person Identification

Get up and running with the EEG person identification project in minutes!

## Step-by-Step Instructions

### 1. Install Dependencies (5 minutes)

Open a terminal/command prompt and run:

```bash
cd c:\Users\Administrator\Desktop\EEG_Identification
pip install -r requirements.txt
```

**Note**: This will install all required libraries including TensorFlow, MNE, and scientific computing packages.

### 2. Start Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your web browser.

### 3. Run the Notebooks in Order

#### Notebook 1: Preprocessing (30-60 minutes)

1. Open `1_preprocessing.ipynb`
2. Click **Cell → Run All** or press `Shift+Enter` for each cell
3. The notebook will:
   - Automatically download the PhysioNet dataset
   - Preprocess EEG signals
   - Generate spectrograms
   - Save processed data

**Important**: This notebook will download ~2GB of data and create processed files.

#### Notebook 2: Model Training (1-3 hours)

1. Open `2_model_training.ipynb`
2. Click **Cell → Run All**
3. The notebook will:
   - Build the CNN+RNN model
   - Train on the preprocessed data
   - Save the best model
   - Generate training plots

**Tip**: If you have a GPU, training will be much faster!

#### Notebook 3: Performance Analysis (10-20 minutes)

1. Open `3_performance_report.ipynb`
2. Click **Cell → Run All**
3. The notebook will:
   - Evaluate the trained model
   - Generate confusion matrix
   - Create t-SNE visualizations
   - Provide comprehensive analysis

## Quick Test (For Testing Only)

If you want to test the pipeline quickly without processing all 109 subjects:

### Modify Preprocessing Config

In `1_preprocessing.ipynb`, find this line:
```python
'n_subjects_to_use': 109,  # Use all subjects
```

Change it to:
```python
'n_subjects_to_use': 20,  # Use only 20 subjects for testing
```

This will complete much faster (~10 minutes preprocessing, ~20 minutes training).

## Expected Outputs

After running all notebooks, you'll have:

### Files Created
- `data/processed/preprocessed_data.h5` - Preprocessed EEG data (~500MB-2GB)
- `models/*.keras` - Trained model weights
- `models/*_results.json` - Training results
- `figures/*.png` - Multiple visualization plots

### Key Visualizations
1. Raw EEG signals
2. Spectrograms (time-frequency representations)
3. Training curves (loss, accuracy)
4. Confusion matrix (109×109)
5. Per-subject performance
6. t-SNE feature embeddings
7. Error analysis plots

## Performance Expectations

### With Full Dataset (109 subjects)
- **Test Accuracy**: 70-85%
- **Top-5 Accuracy**: 90-95%
- **Training Time**: 1-3 hours (GPU) / 4-6 hours (CPU)

### With Reduced Dataset (20 subjects for testing)
- **Test Accuracy**: 85-95% (easier task)
- **Training Time**: 15-30 minutes (GPU) / 1-2 hours (CPU)

## Troubleshooting

### Issue: "No module named 'mne'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "Memory Error"
**Solution**: Reduce the number of subjects:
```python
'n_subjects_to_use': 50,  # Instead of 109
```

### Issue: Training is very slow
**Solution**:
- Check if GPU is detected: In notebook, run `tf.config.list_physical_devices('GPU')`
- If no GPU, reduce batch size or use fewer subjects for testing

### Issue: "FileNotFoundError" for model
**Solution**: Make sure you ran notebooks 1 and 2 before notebook 3

## GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU:

1. **Install CUDA Toolkit** (if not already installed)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Version 11.8 or 12.x recommended

2. **Verify GPU is detected**:
   In a notebook cell, run:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

3. **Expected output**:
   ```
   [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```

## Understanding the Results

### Key Metrics

1. **Accuracy**: Percentage of correctly identified subjects
   - Good: >80%
   - Excellent: >90%

2. **F1-Score**: Balance between precision and recall
   - Range: 0-1 (higher is better)
   - Good: >0.75

3. **Top-5 Accuracy**: Is the correct subject in the top 5 predictions?
   - Usually 10-15% higher than top-1 accuracy

### Interpreting Visualizations

1. **Confusion Matrix**:
   - Diagonal = correct predictions (bright green)
   - Off-diagonal = confusions between subjects
   - Look for patterns: which subjects are confused?

2. **t-SNE Plot**:
   - Clusters = subjects with similar EEG patterns
   - Well-separated = easier to identify
   - Overlapping = more challenging

3. **Training Curves**:
   - Should show decreasing loss
   - Validation accuracy should plateau
   - If validation loss increases while training loss decreases = overfitting

## Next Steps

After running all notebooks:

1. **Review Results**: Check `figures/` directory for all plots
2. **Read Discussion**: Open `figures/performance_discussion.txt`
3. **Experiment**: Try different model architectures or preprocessing parameters
4. **Improve**: Implement data augmentation or attention mechanisms

## Tips for Best Results

1. **Use GPU**: Training is 3-5x faster with GPU
2. **Be Patient**: First run takes time due to data download and preprocessing
3. **Monitor Progress**: Watch the progress bars in notebooks
4. **Save Work**: Notebooks auto-save, but manually save important results
5. **Check Logs**: If errors occur, read the error messages carefully

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the main [README.md](README.md) for detailed information
3. Verify all dependencies are installed correctly
4. Make sure you have sufficient disk space (10GB+) and RAM (8GB+)

## Summary Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run `1_preprocessing.ipynb` (creates processed data)
- [ ] Run `2_model_training.ipynb` (trains model)
- [ ] Run `3_performance_report.ipynb` (analyzes results)
- [ ] Check `figures/` directory for visualizations
- [ ] Review performance metrics in notebook outputs

## Estimated Total Time

| Configuration | Time |
|---------------|------|
| Full dataset + GPU | 2-3 hours |
| Full dataset + CPU | 6-8 hours |
| 20 subjects + GPU | 30-45 minutes |
| 20 subjects + CPU | 2-3 hours |

---

**Ready to start?** Open Jupyter and begin with `1_preprocessing.ipynb`!

Good luck with your EEG person identification project!
