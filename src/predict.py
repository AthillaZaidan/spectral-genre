"""
Predict genre for a single audio file
"""

import sys
import numpy as np
import joblib
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import loadAudioFile, extractAllFeatures

def predictGenre(audioPath, modelPath='../models/genre_classifier.pkl'):
    """
    Predict genre of an audio file
    
    Args:
        audioPath: Path to audio file
        modelPath: Path to trained model
    
    Returns:
        Predicted genre
    """
    print(f"Loading model from {modelPath}...")
    modelPackage = joblib.load(modelPath)
    
    scaler = modelPackage['scaler']
    selector = modelPackage['selector']
    model = modelPackage['model']
    genres = modelPackage['genres']
    
    print(f"Loading audio from {audioPath}...")
    audio, sr = loadAudioFile(audioPath)
    
    if audio is None:
        print("Error loading audio file")
        return None
    
    print("Extracting features...")
    features = extractAllFeatures(audio)
    features = features.reshape(1, -1)
    
    print("Preprocessing...")
    featuresScaled = scaler.transform(features)
    featuresSelected = selector.transform(featuresScaled)
    
    print("Predicting...")
    prediction = model.predict(featuresSelected)[0]
    
    probabilities = model.decision_function(featuresSelected)[0]
    confidence = np.max(probabilities)
    
    print("\n" + "="*50)
    print(f"Predicted Genre: {prediction.upper()}")
    print(f"Confidence Score: {confidence:.2f}")
    print("="*50)
    
    return prediction


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    audioPath = sys.argv[1]
    predictGenre(audioPath)
