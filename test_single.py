"""
Test model on a single file with detailed output
"""

import sys
import numpy as np
import joblib
from utils import loadAudioFile, extractAllFeatures

def testSingleFile(audioPath, modelPath='genre_classifier.pkl'):
    """Test model on single file with detailed analysis"""
    
    print("="*60)
    print("SINGLE FILE TEST")
    print("="*60)
    
    modelPackage = joblib.load(modelPath)
    scaler = modelPackage['scaler']
    selector = modelPackage['selector']
    model = modelPackage['model']
    genres = modelPackage['genres']
    
    print(f"\nFile: {audioPath}")
    
    audio, sr = loadAudioFile(audioPath)
    if audio is None:
        return
    
    print(f"Duration: {len(audio)/sr:.1f}s")
    print(f"Sample Rate: {sr} Hz")
    
    features = extractAllFeatures(audio)
    print(f"Features extracted: {len(features)}")
    
    features = features.reshape(1, -1)
    featuresScaled = scaler.transform(features)
    featuresSelected = selector.transform(featuresScaled)
    print(f"Features after selection: {featuresSelected.shape[1]}")
    
    prediction = model.predict(featuresSelected)[0]
    decisionScores = model.decision_function(featuresSelected)[0]
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Predicted Genre: {prediction.upper()}")
    print("\nDecision Scores (all genres):")
    
    scoreDict = list(zip(genres, decisionScores))
    scoreDict.sort(key=lambda x: x[1], reverse=True)
    
    for genre, score in scoreDict:
        bar = "█" * int(abs(score))
        marker = ">>> " if genre == prediction else "    "
        print(f"{marker}{genre:12s}: {score:7.2f} {bar}")
    
    print("="*60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single.py <audio_file_path>")
        sys.exit(1)
    
    testSingleFile(sys.argv[1])
