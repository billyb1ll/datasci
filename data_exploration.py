#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def extract_audio_features(file_path):
    """
    สกัดคุณลักษณะพื้นฐานจากไฟล์เสียง
    
    Parameters:
        file_path (str): เส้นทางไปยังไฟล์เสียง
        
    Returns:
        features (dict): พจนานุกรมที่มีคุณลักษณะต่าง ๆ
    """
    try:
        # โหลดไฟล์เสียง
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        # สกัดคุณลักษณะ
        # 1. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        
        # 2. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        
        # 3. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # 4. Zero Crossing Rate
        zero_crossing = librosa.feature.zero_crossing_rate(y).mean()
        
        # 5. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y).mean()
        
        # 7. Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # รวมคุณลักษณะในพจนานุกรม
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'rolloff': rolloff,
            'zero_crossing_rate': zero_crossing,
            'rms': rms,
            'tempo': tempo
        }
        
        # เพิ่ม MFCC
        for i, val in enumerate(mfcc_mean):
            features[f'mfcc_{i+1}'] = val
            
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def analyze_dataset(dataset_path, genres, num_samples=None):
    """
    วิเคราะห์ชุดข้อมูลและแสดงภาพเชิงลึก
    
    Parameters:
        dataset_path (str): เส้นทางไปยังโฟลเดอร์ที่มีไฟล์เส