"""
Quick training script (no CV, just hold-out test)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import loadAudioFile, extractAllFeatures

def main():
    print("QUICK TRAINING MODE")
    print("="*60)
    
    datasetPath = '../data/gtzan/genres_original'
    genreList = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("\nLoading dataset...")
    audioList = []
    labelList = []
    
    for genre in genreList:
        genrePath = Path(datasetPath) / genre
        for audioFile in genrePath.glob('*.wav'):
            audio, sr = loadAudioFile(str(audioFile))
            if audio is not None:
                audioList.append(audio)
                labelList.append(genre)
    
    print(f"Loaded {len(audioList)} files")
    
    print("\nExtracting features...")
    featureMatrix = [extractAllFeatures(audio) for audio in tqdm(audioList)]
    featureMatrix = np.array(featureMatrix)
    
    print("\nTraining...")
    xTrain, xTest, yTrain, yTest = train_test_split(
        featureMatrix, labelList, test_size=0.2, stratify=labelList, random_state=42
    )
    
    scaler = RobustScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    selector = SelectKBest(score_func=mutual_info_classif, k=100)
    xTrainSelected = selector.fit_transform(xTrainScaled, yTrain)
    xTestSelected = selector.transform(xTestScaled)
    
    model = SVC(C=100, gamma=0.01, kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(xTrainSelected, yTrain)
    
    yPred = model.predict(xTestSelected)
    accuracy = accuracy_score(yTest, yPred)
    
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    modelPackage = {
        'scaler': scaler,
        'selector': selector,
        'model': model,
        'genres': genreList
    }
    import os
    os.makedirs('../models', exist_ok=True)
    joblib.dump(modelPackage, '../models/genre_classifier_fast.pkl')
    print("Saved: ../models/genre_classifier_fast.pkl")


if __name__ == '__main__':
    main()
