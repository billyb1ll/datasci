#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import os
import librosa
from tensorflow.keras.models import load_model

# Import our existing feature extraction function
from main import extract_features

def load_test_data():
    """
    Load the test data and labels.
    If you've already saved the preprocessed test data, load it.
    Otherwise, you'll need to process the raw audio files again.
    
    Returns:
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder object
    """
    # Check if processed test data exists
    try:
        # Try to load preprocessed data
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        # Load label encoder classes
        label_encoder_classes = np.load('data/label_encoder_classes.npy', allow_pickle=True)
        
        # Recreate label encoder with the saved classes
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        
        print("Loaded preprocessed test data")
        
    except FileNotFoundError:
        print("Preprocessed test data not found. Please run main.py first to generate the test data.")
        return None, None, None
    
    return X_test, y_test, label_encoder

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model on test data
    
    Parameters:
        model: Loaded Keras model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder used to transform genre names
    """
    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Generate classification report
    genre_names = label_encoder.classes_
    class_report = classification_report(y_test, y_pred, target_names=genre_names)
    print("\nClassification Report:")
    print(class_report)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_names, 
                yticklabels=genre_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, precision, recall, f1

def predict_genre(model, audio_path, label_encoder):
    """
    Predict the genre of an audio file
    
    Parameters:
        model: Loaded Keras model
        audio_path: Path to the audio file
        label_encoder: Label encoder used to transform genre names
    
    Returns:
        predicted_genre: Predicted genre name
        confidence: Prediction confidence
    """
    # Extract features from the audio file
    mel_spectrogram = extract_features(audio_path)
    
    # Reshape for model input
    # Make sure the shape matches what the model expects
    mel_spectrogram = mel_spectrogram[:, :259]  # Ensure same width as training data
    X = mel_spectrogram.reshape(1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)
    
    # Make prediction
    predictions = model.predict(X)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # Convert class index to genre name
    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_genre, confidence, predictions[0]

if __name__ == "__main__":
    # Path to the saved model
    model_path = 'crnn_music_genre_model_best.keras'
    
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained and saved the model using main.py")
        exit(1)
    
    # Load test data
    X_test, y_test, label_encoder = load_test_data()
    
    if X_test is not None:
        # Evaluate model on test data
        print("\nEvaluating model on test data...")
        evaluate_model(model, X_test, y_test, label_encoder)
    
    # Example: Predict genre for a specific audio file
    print("\nTesting prediction on a sample file...")
    
    # Look for a test audio file
    dataset_path = '/home/bill/code/AI/data/Data/genres_original'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Get the first audio file from a random genre
    import random
    test_genre = random.choice(genres)
    genre_path = os.path.join(dataset_path, test_genre)
    
    try:
        test_files = [f for f in os.listdir(genre_path) if f.endswith('.wav') or f.endswith('.mp3')]
        if test_files:
            test_file = os.path.join(genre_path, test_files[0])
            print(f"Testing with file: {test_file} (actual genre: {test_genre})")
            
            # Predict
            predicted_genre, confidence, all_probs = predict_genre(model, test_file, label_encoder)
            print(f"Predicted genre: {predicted_genre} with {confidence:.2f}% confidence")
            
            # Show top 3 predictions
            top_indices = np.argsort(-all_probs)[:3]
            print("\nTop 3 predictions:")
            for idx in top_indices:
                genre = label_encoder.inverse_transform([idx])[0]
                prob = all_probs[idx] * 100
                print(f"{genre}: {prob:.2f}%")
        else:
            print(f"No audio files found in {genre_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure your dataset path is correct and contains audio files.")
