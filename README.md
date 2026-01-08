# Spectral Genre Classification

A music genre classification system using Singular Value Decomposition (SVD) and Machine Learning techniques to classify audio files into 10 different genres.

## Overview

This project implements a machine learning pipeline that analyzes audio files and classifies them into one of ten music genres. It leverages advanced audio feature extraction techniques including SVD decomposition of spectrograms, Mel-Frequency Cepstral Coefficients (MFCC), and various spectral features. The classification is performed using a Support Vector Machine (SVM) classifier with robust preprocessing and feature selection.

## Project Structure

```
spectral-genre/
├── src/                      # Python source code
│   ├── train.py              # Complete training script (with cross-validation)
│   ├── train_fast.py         # Fast training script (without cross-validation)
│   ├── predict.py            # Genre prediction for audio files
│   ├── test_single.py        # Detailed model testing on single files
│   └── utils.py              # Utility functions for feature extraction
│
├── data/                     # Dataset directory
│   └── gtzan/
│       └── genres_original/  # Audio files in WAV format
│           ├── blues/
│           ├── classical/
│           ├── country/
│           ├── disco/
│           ├── hiphop/
│           ├── jazz/
│           ├── metal/
│           ├── pop/
│           ├── reggae/
│           └── rock/
│
├── models/                   # Trained model files (*.pkl)
│   └── genre_classifier.pkl
│
├── results/                  # Training results and visualizations
│   ├── confusion_matrix.png
│   └── training_results.csv
│
├── notebooks/                # Jupyter notebooks for analysis
│   └── spectra.ipynb
│
├── cache.npz                 # Feature cache file
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository and navigate to the project directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

**Complete training with cross-validation:**

This method provides the most accurate evaluation of model performance using stratified k-fold cross-validation.

```bash
cd src
python train.py
```

**Outputs:**
- `../models/genre_classifier.pkl` - Trained model with scaler, feature selector, and SVM classifier
- `../results/confusion_matrix.png` - Visualization of classification performance
- `../results/training_results.csv` - Detailed training metrics and results

**Fast training without cross-validation:**

For quick prototyping or when time is limited, use the fast training mode.

```bash
cd src
python train_fast.py
```

**Output:**
- `../models/genre_classifier_fast.pkl` - Trained model using simple train/test split

### 2. Predicting Genre for Audio Files

Classify a single audio file:

```bash
cd src
python predict.py path/to/audio.wav
```

The script will load the trained model, extract features from the audio file, and output the predicted genre.

### 3. Detailed Testing on Single Files

Test the model on a single file with comprehensive output including confidence scores:

```bash
cd src
python test_single.py path/to/audio.wav
```

This provides detailed information including:
- Audio file metadata (duration, sample rate)
- Feature extraction details
- Prediction results with confidence scores for all genres

## Features Extracted

The system extracts a comprehensive set of 164 features from each audio file:

### SVD Features (58 features)
- **50 singular values** extracted from the STFT (Short-Time Fourier Transform) matrix
- **8 statistical measures**: mean, standard deviation, max, min, median, Q1, Q3, and energy ratio

### MFCC Features (80 features)
- **20 MFCC coefficients** with their statistical measures
- Mean and standard deviation for each coefficient
- First and second-order delta features (velocity and acceleration)

### Spectral Features (26 features)
- Spectral centroid (mean and standard deviation)
- Spectral bandwidth (mean and standard deviation)
- Spectral rolloff (mean and standard deviation)
- Spectral contrast across 7 frequency bands (mean and standard deviation)
- Spectral flatness (mean and standard deviation)
- Root Mean Square energy (mean and standard deviation)
- Zero Crossing Rate (mean and standard deviation)

## Machine Learning Pipeline

1. **Feature Extraction**: Audio signals are processed to extract 164 features
2. **Preprocessing**: Features are scaled using RobustScaler (resistant to outliers)
3. **Feature Selection**: SelectKBest with mutual information selects the most informative features
4. **Classification**: Support Vector Machine (SVM) with RBF kernel performs the final classification
5. **Evaluation**: Models are evaluated using stratified cross-validation and confusion matrices

## Dataset

This project uses the GTZAN Genre Collection dataset, which contains:
- **1,000 audio tracks** (100 tracks per genre)
- **10 genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Duration**: Each track is 30 seconds long
- **Format**: WAV files, 22,050 Hz sample rate

## Output Directories

All outputs are automatically saved to their respective directories:

- **models/**: Serialized model files (.pkl) containing the trained classifier, scaler, and feature selector
- **results/**: Confusion matrices, training metrics, and visualization files
- **src/**: All Python source code
- **data/**: Audio dataset files

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```
