# Spectral Genre Classification

Klasifikasi genre musik menggunakan SVD dan Machine Learning.

## Struktur Folder

```
spectral-genre/
├── src/                      # Source code Python
│   ├── train.py              # Training script lengkap (dengan CV)
│   ├── train_fast.py         # Training script cepat (tanpa CV)
│   ├── predict.py            # Prediksi genre untuk file audio
│   ├── test_single.py        # Test model dengan detail
│   └── utils.py              # Utility functions (feature extraction)
│
├── data/                     # Dataset
│   └── gtzan/
│       └── genres_original/  # Audio files (.wav)
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
├── models/                   # Trained models (.pkl)
│   └── genre_classifier.pkl
│
├── results/                  # Training results & visualizations
│   ├── confusion_matrix.png
│   └── training_results.csv
│
├── notebooks/                # Jupyter notebooks
│   └── spectra.ipynb
│
├── cache.npz                 # Cache file
└── requirements.txt          # Dependencies
```

## Cara Menggunakan

### 1. Training Model

**Training lengkap (dengan Cross-Validation):**
```bash
cd src
python train.py
```
Output:
- `../models/genre_classifier.pkl` - Model terlatih
- `../results/confusion_matrix.png` - Visualization
- `../results/training_results.csv` - Hasil training

**Training cepat (tanpa CV):**
```bash
cd src
python train_fast.py
```
Output:
- `../models/genre_classifier_fast.pkl`

### 2. Prediksi Genre

```bash
cd src
python predict.py path/to/audio.wav
```

### 3. Test Single File

```bash
cd src
python test_single.py path/to/audio.wav
```

## Output Locations

Semua output otomatis disimpan di folder yang sesuai:

- **Models**: `models/` - Semua model (.pkl files)
- **Results**: `results/` - Confusion matrix, training results, visualizations
- **Source**: `src/` - Semua source code Python
- **Data**: `data/` - Dataset audio files

## Features

Script ini menggunakan kombinasi fitur:
- **SVD Features**: 50 singular values + statistics
- **MFCC Features**: 20 coefficients dengan deltas
- **Spectral Features**: Centroid, bandwidth, rolloff, contrast, flatness, RMS, ZCR

Total: 164 features

## Dependencies

Install dengan:
```bash
pip install -r requirements.txt
```
