"""
Feature extraction utilities for music genre classification
"""

import numpy as np
import librosa
from scipy.linalg import svd

def extractSvdFeatures(audioSignal, numComponents=50, sampleRate=22050):
    """
    Extract SVD features from STFT
    
    Args:
        audioSignal: Audio time series
        numComponents: Number of singular values to extract
        sampleRate: Sample rate of audio
    
    Returns:
        Array of SVD features (singular values + statistics)
    """
    try:
        stftMatrix = librosa.stft(audioSignal, n_fft=2048, hop_length=512, window='hann')
        
        realPart = stftMatrix.real
        imagPart = stftMatrix.imag
        combinedMatrix = np.vstack([realPart, imagPart])
        
        U, singularValues, Vt = svd(combinedMatrix, full_matrices=False)
        
        topSingularValues = singularValues[:numComponents]
        
        svdMean = np.mean(topSingularValues)
        svdStd = np.std(topSingularValues)
        svdMax = np.max(topSingularValues)
        svdMin = np.min(topSingularValues)
        svdMedian = np.median(topSingularValues)
        svdQ1 = np.percentile(topSingularValues, 25)
        svdQ3 = np.percentile(topSingularValues, 75)
        
        totalEnergy = np.sum(singularValues[:100]) if len(singularValues) >= 100 else np.sum(singularValues)
        energyRatio = np.sum(topSingularValues) / totalEnergy if totalEnergy > 0 else 0
        
        svdStats = [svdMean, svdStd, svdMax, svdMin, svdMedian, svdQ1, svdQ3, energyRatio]
        
        return np.concatenate([topSingularValues, svdStats])
        
    except Exception as e:
        print(f"Error extracting SVD features: {e}")
        return np.zeros(numComponents + 8)


def extractMfccFeatures(audioSignal, numCoeffs=20, sampleRate=22050):
    """
    Extract MFCC features with deltas
    
    Args:
        audioSignal: Audio time series
        numCoeffs: Number of MFCC coefficients
        sampleRate: Sample rate of audio
    
    Returns:
        Array of MFCC features (mean, std, deltas)
    """
    featureList = []
    
    try:
        mfccMatrix = librosa.feature.mfcc(y=audioSignal, sr=sampleRate, n_mfcc=numCoeffs)
        
        mfccMean = np.mean(mfccMatrix, axis=1)
        mfccStd = np.std(mfccMatrix, axis=1)
        featureList.extend(mfccMean)
        featureList.extend(mfccStd)
        
        mfccDelta = librosa.feature.delta(mfccMatrix)
        mfccDeltaMean = np.mean(mfccDelta, axis=1)
        featureList.extend(mfccDeltaMean)
        
        mfccDelta2 = librosa.feature.delta(mfccMatrix, order=2)
        mfccDelta2Mean = np.mean(mfccDelta2, axis=1)
        featureList.extend(mfccDelta2Mean)
        
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        featureList = [0] * (numCoeffs * 4)
    
    return np.array(featureList)


def extractSpectralFeatures(audioSignal, sampleRate=22050):
    """
    Extract spectral features
    
    Args:
        audioSignal: Audio time series
        sampleRate: Sample rate of audio
    
    Returns:
        Array of spectral features
    """
    featureList = []
    
    try:
        spectralCentroid = librosa.feature.spectral_centroid(y=audioSignal, sr=sampleRate)
        featureList.append(np.mean(spectralCentroid))
        featureList.append(np.std(spectralCentroid))
        
        spectralBandwidth = librosa.feature.spectral_bandwidth(y=audioSignal, sr=sampleRate)
        featureList.append(np.mean(spectralBandwidth))
        featureList.append(np.std(spectralBandwidth))
        
        spectralRolloff = librosa.feature.spectral_rolloff(y=audioSignal, sr=sampleRate)
        featureList.append(np.mean(spectralRolloff))
        featureList.append(np.std(spectralRolloff))
        
        spectralContrast = librosa.feature.spectral_contrast(y=audioSignal, sr=sampleRate)
        contrastMean = np.mean(spectralContrast, axis=1)
        contrastStd = np.std(spectralContrast, axis=1)
        featureList.extend(contrastMean)
        featureList.extend(contrastStd)
        
        spectralFlatness = librosa.feature.spectral_flatness(y=audioSignal)
        featureList.append(np.mean(spectralFlatness))
        featureList.append(np.std(spectralFlatness))
        
        rmsEnergy = librosa.feature.rms(y=audioSignal)
        featureList.append(np.mean(rmsEnergy))
        featureList.append(np.std(rmsEnergy))
        
        zeroCrossingRate = librosa.feature.zero_crossing_rate(audioSignal)
        featureList.append(np.mean(zeroCrossingRate))
        featureList.append(np.std(zeroCrossingRate))
        
    except Exception as e:
        print(f"Error extracting spectral features: {e}")
        featureList = [0] * 26
    
    return np.array(featureList)


def extractAllFeatures(audioSignal, sampleRate=22050):
    """
    Extract all features (SVD + MFCC + Spectral)
    
    Args:
        audioSignal: Audio time series
        sampleRate: Sample rate of audio
    
    Returns:
        Combined feature vector (164 features)
    """
    svdFeatures = extractSvdFeatures(audioSignal, numComponents=50, sampleRate=sampleRate)
    mfccFeatures = extractMfccFeatures(audioSignal, numCoeffs=20, sampleRate=sampleRate)
    spectralFeatures = extractSpectralFeatures(audioSignal, sampleRate=sampleRate)
    
    allFeatures = np.concatenate([svdFeatures, mfccFeatures, spectralFeatures])
    
    return allFeatures


def loadAudioFile(filePath, sampleRate=22050, duration=30):
    """
    Load audio file
    
    Args:
        filePath: Path to audio file
        sampleRate: Target sample rate
        duration: Duration to load (seconds)
    
    Returns:
        Audio signal and sample rate
    """
    try:
        audioSignal, sr = librosa.load(filePath, sr=sampleRate, duration=duration)
        return audioSignal, sr
    except Exception as e:
        print(f"Error loading audio file {filePath}: {e}")
        return None, None
