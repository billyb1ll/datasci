import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
import tempfile
from collections import Counter

# Constants (Must match the values used in training)
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_MEL_LENGTH = 1292  # Ensure consistent Mel-spectrogram size

# Load the trained model
model_path = "music_classification_model.keras"  # Change path if needed
model = load_model(model_path)
print(f"✅ Loaded model from {model_path}")

# Load label encoder (Manually define or load from training phase)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']  # Change if different

label_encoder = LabelEncoder()
label_encoder.fit(GENRES)


def extract_mel_spectrogram(file_path):
    """Extract Mel-Spectrogram from the input audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Ensure the audio has the same length
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        elif len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))

        # Compute Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                                  hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure fixed size
        if mel_spec_db.shape[1] > FIXED_MEL_LENGTH:
            mel_spec_db = mel_spec_db[:, :FIXED_MEL_LENGTH]
        elif mel_spec_db.shape[1] < FIXED_MEL_LENGTH:
            pad_width = ((0, 0), (0, FIXED_MEL_LENGTH - mel_spec_db.shape[1]))
            mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')

        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def convert_audio(file_path, target_format="wav", target_sample_rate=SAMPLE_RATE):
    """
    Convert audio file to target format and sample rate
    Returns path to the converted file
    """
    try:
        # Get file extension from the input file
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension[1:].lower()  # Remove the dot and make lowercase
        
        # Load audio using pydub (handles various formats)
        if file_extension == "mp3":
            audio = AudioSegment.from_mp3(file_path)
        elif file_extension == "wav":
            audio = AudioSegment.from_wav(file_path)
        elif file_extension == "ogg":
            audio = AudioSegment.from_ogg(file_path)
        elif file_extension in ["m4a", "mp4"]:
            audio = AudioSegment.from_file(file_path, format="mp4")
        elif file_extension == "flac":
            audio = AudioSegment.from_file(file_path, format="flac")
        else:
            # Try generic approach
            audio = AudioSegment.from_file(file_path)
        
        # Set the sample rate if needed
        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
        
        # Create temporary file for the converted audio
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{target_format}", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Export to the target format
        audio.export(temp_file_path, format=target_format)
        
        print(f"✅ Converted {file_path} to {target_format} format at {target_sample_rate}Hz")
        return temp_file_path
        
    except Exception as e:
        print(f"❌ Error converting {file_path}: {e}")
        return file_path  # Return original file path if conversion fails


def segment_audio(audio_data, sr=SAMPLE_RATE, segment_duration=DURATION):
    """
    Segment audio data into chunks of specified duration
    Returns a list of audio segments
    """
    segment_length = sr * segment_duration
    total_samples = len(audio_data)
    
    # Calculate number of complete segments
    num_segments = total_samples // segment_length
    
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio_data[start:end]
        segments.append(segment)
    
    # Add final segment if it's at least half the segment_duration
    remaining_samples = total_samples % segment_length
    if remaining_samples >= segment_length // 2:
        final_segment = audio_data[-segment_length:]  # Take the last segment_length samples
        segments.append(final_segment)
    
    print(f"✅ Segmented audio into {len(segments)} segments of {segment_duration} seconds each")
    return segments

def predict_genre(file_path, is_full_song=False):
    """Predict the genre of an input audio file."""
    # Convert the audio file to proper format first
    converted_file = convert_audio(file_path)
    
    if not is_full_song:
        # Process as a single clip (original behavior)
        mel_spectrogram = extract_mel_spectrogram(converted_file)
        
        # Clean up temporary file if it was created
        if converted_file != file_path and os.path.exists(converted_file):
            os.remove(converted_file)
            
        if mel_spectrogram is None:
            print("❌ Failed to extract features.")
            return

        # Prepare input for the model
        mel_spectrogram = mel_spectrogram.reshape(1, N_MELS, FIXED_MEL_LENGTH, 1)  # Add batch & channel dimensions
        mel_spectrogram = np.repeat(mel_spectrogram, 3, axis=-1)  # Convert to 3-channel if model requires

        # Run prediction
        predictions = model.predict(mel_spectrogram)
        predicted_label = np.argmax(predictions)
        predicted_genre = label_encoder.inverse_transform([predicted_label])[0]
        
        # Convert to percentages
        percentages = predictions[0] * 100

        print(f"🎵 Predicted Genre: {predicted_genre}")
        print("\nGenre Percentages:")
        for i, genre in enumerate(GENRES):
            print(f"{genre}: {percentages[i]:.2f}%")
        
        # Display the probability for each genre as percentages
        plt.figure(figsize=(10, 4))
        bars = plt.bar(GENRES, percentages, color='skyblue')
        plt.xlabel("Genres")
        plt.ylabel("Percentage (%)")
        plt.title(f"Predicted Genre: {predicted_genre}")
        plt.xticks(rotation=45)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Set y-axis to go from 0 to just above the max percentage (+ 5% padding)
        plt.ylim(0, max(percentages) * 1.1)
        
        plt.tight_layout()
        plt.show()
        
        return predicted_genre
    
    else:
        # Process as a full song by segmenting and analyzing each part
        try:
            # Load entire audio
            y, sr = librosa.load(converted_file, sr=SAMPLE_RATE)
            
            # Clean up temporary file if it was created
            if converted_file != file_path and os.path.exists(converted_file):
                os.remove(converted_file)
                
            # Segment the audio
            segments = segment_audio(y, sr)
            
            if not segments:
                print("❌ Failed to segment audio.")
                return
            
            # Process each segment and collect predictions
            segment_predictions = []
            segment_genres = []
            all_probabilities = np.zeros(len(GENRES))
            
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)}...")
                
                # Convert segment to mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=N_FFT, 
                                                      hop_length=HOP_LENGTH, n_mels=N_MELS)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Ensure fixed size
                if mel_spec_db.shape[1] > FIXED_MEL_LENGTH:
                    mel_spec_db = mel_spec_db[:, :FIXED_MEL_LENGTH]
                elif mel_spec_db.shape[1] < FIXED_MEL_LENGTH:
                    pad_width = ((0, 0), (0, FIXED_MEL_LENGTH - mel_spec_db.shape[1]))
                    mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
                
                # Predict for this segment
                mel_spec_db = mel_spec_db.reshape(1, N_MELS, FIXED_MEL_LENGTH, 1)
                mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
                
                seg_predictions = model.predict(mel_spec_db, verbose=0)
                seg_label = np.argmax(seg_predictions)
                seg_genre = label_encoder.inverse_transform([seg_label])[0]
                
                segment_predictions.append(seg_predictions[0])
                segment_genres.append(seg_genre)
                all_probabilities += seg_predictions[0]
            
            # Aggregate results
            # Method 1: Voting (most common genre)
            genre_votes = Counter(segment_genres)
            most_common_genre, vote_count = genre_votes.most_common(1)[0]
            vote_percentage = (vote_count / len(segments)) * 100
            
            # Method 2: Average probabilities
            avg_probabilities = all_probabilities / len(segments)
            avg_label = np.argmax(avg_probabilities)
            avg_genre = label_encoder.inverse_transform([avg_label])[0]
            
            # Convert to percentages
            avg_percentages = avg_probabilities * 100
            
            # Display results
            print(f"\n🎵 Full Song Analysis Results:")
            print(f"✓ Most voted genre: {most_common_genre} ({vote_count}/{len(segments)} segments, {vote_percentage:.1f}%)")
            print(f"✓ Highest average probability genre: {avg_genre}")
            
            print("\nGenre Percentages:")
            for i, genre in enumerate(GENRES):
                print(f"{genre}: {avg_percentages[i]:.2f}%")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot segment-by-segment predictions as percentages
            plt.subplot(2, 1, 1)
            segment_indices = list(range(len(segments)))
            for g_idx, genre in enumerate(GENRES):
                genre_percentages = [seg_pred[g_idx] * 100 for seg_pred in segment_predictions]
                plt.plot(segment_indices, genre_percentages, label=genre)
            
            plt.xlabel("Segment Index")
            plt.ylabel("Percentage (%)")
            plt.title("Genre Percentages Across Song Segments")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot overall average percentages
            plt.subplot(2, 1, 2)
            bars = plt.bar(GENRES, avg_percentages, color='skyblue')
            plt.xlabel("Genres")
            plt.ylabel("Percentage (%)")
            plt.title(f"Overall Song Genre: {avg_genre}")
            plt.xticks(rotation=45)
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Set y-axis to go from 0 to just above the max percentage (+ 5% padding)
            plt.ylim(0, max(avg_percentages) * 1.1)
            
            plt.tight_layout()
            plt.show()
            
            return avg_genre  # Return the genre with highest average probability
        
        except Exception as e:
            print(f"❌ Error processing full song: {e}")
            return None


# Example usage
audio_file = "/home/bill/code/AI/rock.00001.wav"  # Change this to your audio file path
# For single clip analysis:
predict_genre(audio_file)

# For full song analysis:
# full_song = "Baby One More Time - Britney Spears.mp3"
# predict_genre(full_song, is_full_song=True)
