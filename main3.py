import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
# Define fixed sizes for features to ensure consistent dimensions
FIXED_MEL_LENGTH = 1292  # This is approximately 30 seconds at 22050Hz sampling rate

def augment_audio(y, sr=SAMPLE_RATE):
    """Apply random augmentations to audio signal"""
    # Choose which augmentations to apply (randomly)
    augmentations = []
    if np.random.random() > 0.5:
        augmentations.append('time_stretch')
    if np.random.random() > 0.5:
        augmentations.append('pitch_shift')
    if np.random.random() > 0.5:
        augmentations.append('add_noise')
    if np.random.random() > 0.5:
        augmentations.append('time_shift')
    
    # If no augmentation was selected, pick one randomly
    if not augmentations:
        augmentations.append(np.random.choice(['time_stretch', 'pitch_shift', 'add_noise', 'time_shift']))
    
    # Apply selected augmentations
    y_augmented = y.copy()
    
    for aug in augmentations:
        if aug == 'time_stretch':
            # Random time stretching (0.8 to 1.2 times speed)
            rate = np.random.uniform(0.8, 1.2)
            y_augmented = librosa.effects.time_stretch(y_augmented, rate=rate)
        
        elif aug == 'pitch_shift':
            # Random pitch shifting (-3 to 3 semitones)
            n_steps = np.random.uniform(-3, 3)
            y_augmented = librosa.effects.pitch_shift(y_augmented, sr=sr, n_steps=n_steps)
        
        elif aug == 'add_noise':
            # Add random noise (SNR between 10 and 20 dB)
            noise_factor = np.random.uniform(0.005, 0.02)
            noise = np.random.randn(len(y_augmented))
            y_augmented = y_augmented + noise_factor * noise
        
        elif aug == 'time_shift':
            # Random time shifting (up to 15% of the signal)
            shift_factor = int(np.random.uniform(-0.15, 0.15) * len(y_augmented))
            y_augmented = np.roll(y_augmented, shift_factor)
            
            # If rolling brings in values from the end to the beginning, or vice versa, zero them out
            if shift_factor > 0:
                y_augmented[:shift_factor] = 0
            else:
                y_augmented[shift_factor:] = 0
    
    # Ensure the audio length is standardized
    if len(y_augmented) > SAMPLES_PER_TRACK:
        y_augmented = y_augmented[:SAMPLES_PER_TRACK]
    elif len(y_augmented) < SAMPLES_PER_TRACK:
        y_augmented = np.pad(y_augmented, (0, SAMPLES_PER_TRACK - len(y_augmented)))
    
    return y_augmented

def download_gtzan_dataset():
    """Download GTZAN dataset if not exists"""
    dataset_path = os.path.join(os.path.expanduser("~"), "datasets", "gtzan")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        print("Downloading GTZAN dataset...")
        import urllib.request
        import tarfile
        url = "https://opihi.cs.uvic.ca/sound/genres.tar.gz"
        tar_path = os.path.join(dataset_path, "genres.tar.gz")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract tar file
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=dataset_path)
        print(f"Dataset downloaded and extracted to {dataset_path}")
    else:
        print(f"Dataset already exists at {dataset_path}")
    return os.path.join(dataset_path, "genres")

def extract_features(file_path, augment=False):
    """Extract MFCC, Mel-spectrogram, chord, key, and BPM from audio file with optional augmentation"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Ensure the audio length is standardized
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        elif len(y) < SAMPLES_PER_TRACK:
            # Pad with zeros if audio is shorter than expected
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        
        # Apply augmentation if requested
        if augment:
            y = augment_audio(y, sr)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Mel-spectrogram with fixed size
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                                        hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure mel-spectrogram has consistent size
        if mel_spectrogram_db.shape[1] > FIXED_MEL_LENGTH:
            mel_spectrogram_db = mel_spectrogram_db[:, :FIXED_MEL_LENGTH]
        elif mel_spectrogram_db.shape[1] < FIXED_MEL_LENGTH:
            # Pad with zeros if spectrogram is shorter than expected
            pad_width = ((0, 0), (0, FIXED_MEL_LENGTH - mel_spectrogram_db.shape[1]))
            mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width, mode='constant')
        
        # Chromagram for chord estimation
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chord_feature = np.mean(chroma.T, axis=0)
        
        # Key estimation
        # Using the chromagram to estimate key (simplified approach)
        key_feature = np.argmax(np.sum(chroma, axis=1))
        
        # BPM estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        return {
            'mfcc': mfcc_scaled,
            'mel_spectrogram': mel_spectrogram_db,
            'chord': chord_feature,
            'key': key_feature,
            'bpm': tempo
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_dataset(dataset_path, apply_augmentation=False, augmentation_factor=1):
    """Process all files in the dataset and extract features, with optional augmentation"""
    genres = os.listdir(dataset_path)
    genres = [g for g in genres if os.path.isdir(os.path.join(dataset_path, g))]
    
    features = []
    labels = []
    
    for genre_idx, genre in enumerate(genres):
        genre_path = os.path.join(dataset_path, genre)
        print(f"Processing {genre} files...")
        
        for filename in os.listdir(genre_path):
            if filename.endswith('.wav') or filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                
                # Extract original features
                extracted_features = extract_features(file_path)
                if extracted_features:
                    features.append(extracted_features)
                    labels.append(genre)
                
                # Apply augmentation if requested
                if apply_augmentation:
                    for i in range(augmentation_factor):
                        augmented_features = extract_features(file_path, augment=True)
                        if augmented_features:
                            features.append(augmented_features)
                            labels.append(genre)
                            print(f"Created augmented sample {i+1} for {filename}")
    
    print(f"Total samples after processing: {len(features)}")
    return features, labels

def prepare_data_for_model(features, labels):
    """Prepare data for CRNN model"""
    # Convert labels to one-hot encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)
    
    # Prepare input features - ensuring all have the same dimensions
    valid_features = []
    valid_labels = []
    
    for i, f in enumerate(features):
        try:
            # Verify the shape is as expected
            if f['mel_spectrogram'].shape == (N_MELS, FIXED_MEL_LENGTH):
                valid_features.append(f)
                valid_labels.append(labels[i])
            else:
                print(f"Skipping sample {i} with unexpected shape: {f['mel_spectrogram'].shape}")
        except Exception as e:
            print(f"Error processing feature {i}: {e}")
    
    print(f"Using {len(valid_features)} valid samples out of {len(features)} total")
    
    # Create arrays from valid features
    X_mel = np.array([f['mel_spectrogram'] for f in valid_features])
    
    # Add channel dimension for CNN
    X_mel = X_mel.reshape(X_mel.shape[0], X_mel.shape[1], X_mel.shape[2], 1)
    
    # Convert to 3-channel for pre-trained models by repeating the same data
    # This converts grayscale spectrograms to RGB-like format
    X_mel_3channel = np.repeat(X_mel, 3, axis=3)
    
    # Get new one-hot labels for valid samples only
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(valid_labels)
    y_onehot = to_categorical(y_encoded)
    
    # Split the data - return both 1-channel and 3-channel versions
    X_train, X_test, y_train, y_test = train_test_split(X_mel_3channel, y_onehot, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder

def build_crnn_model(input_shape, num_classes):
    """Build a CRNN model architecture"""
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Reshape for RNN
    x = layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    
    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def load_pretrained_model(input_shape, num_classes):
    """Load a pre-trained model and adapt it for our task"""
    # Ensure input shape has 3 channels for pre-trained models
    if input_shape[-1] != 3:
        print(f"Warning: Pre-trained models expect 3-channel input but got {input_shape}")
    
    try:
        # Load a pre-trained model (EfficientNetB0 works better for spectrograms)
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print("Loaded EfficientNetB0 model")
    except Exception as e:
        print(f"Error loading EfficientNetB0: {e}, trying ResNet50...")
        # Fallback to ResNet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print("Loaded ResNet50 model")
    
    # Only freeze the first 70% of the base model layers
    # This allows fine-tuning of the deeper layers which are more specialized
    trainable_layers = int(len(base_model.layers) * 0.3)
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    print(f"Made the last {trainable_layers} layers of the base model trainable for fine-tuning")
    
    # Add custom layers for audio classification
    x = base_model.output
    
    # Add spatial dropout for better regularization
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Global average pooling to reduce parameters
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with stronger regularization
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def find_optimal_learning_rate(model, X_train, y_train, min_lr=1e-6, max_lr=1e-2):
    """Learning rate finder to determine the optimal learning rate"""
    # Number of batches to run
    num_batches = 100
    batch_size = 32
    
    # Create learning rate range
    learning_rates = np.geomspace(min_lr, max_lr, num=num_batches)
    losses = []
    
    # Original weights
    original_weights = model.get_weights()
    
    # Train for one batch with each learning rate
    for lr in learning_rates:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Select a random batch
        indices = np.random.randint(0, len(X_train), batch_size)
        history = model.fit(X_train[indices], y_train[indices], epochs=1, verbose=0)
        losses.append(history.history['loss'][0])
    
    # Reset model weights
    model.set_weights(original_weights)
    
    # Find the optimal learning rate (where loss is decreasing the fastest)
    smoothed_losses = []
    for i in range(len(losses)):
        if i == 0:
            smoothed_losses.append(losses[i])
        else:
            smoothed_losses.append(0.9 * smoothed_losses[i-1] + 0.1 * losses[i])
    
    # Calculate gradient
    gradients = np.gradient(smoothed_losses)
    
    # Find optimal learning rate (steepest descent)
    optimal_idx = np.argmin(gradients)
    optimal_lr = learning_rates[optimal_idx]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.plot(learning_rates, smoothed_losses, 'r-', linewidth=2)
    plt.axvline(x=optimal_lr, color='green', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title(f'Learning Rate Finder (Optimal LR: {optimal_lr:.7f})')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "learning_rate_finder.png"))
    
    print(f"Suggested optimal learning rate: {optimal_lr:.7f}")
    return optimal_lr

def create_advanced_data_augmentation():
    """Create a more effective augmentation pipeline for spectrograms"""
    return tf.keras.Sequential([
        # Spatial transformations
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip(mode='horizontal'),  # Horizontal flip is meaningful for spectrograms
        
        # Intensity transformations
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        
        # Custom frequency and time masking (simulating SpecAugment)
        # This lambda performs random frequency masking
        layers.Lambda(lambda x: tf.map_fn(
            lambda img: tf.py_function(
                func=lambda i: apply_frequency_masking(i, max_masks=2, max_width=20),
                inp=[img],
                Tout=tf.float32
            ),
            x
        )),
        
        # This lambda performs random time masking
        layers.Lambda(lambda x: tf.map_fn(
            lambda img: tf.py_function(
                func=lambda i: apply_time_masking(i, max_masks=2, max_width=50),
                inp=[img],
                Tout=tf.float32
            ),
            x
        ))
    ])

def apply_frequency_masking(spectrogram, max_masks=2, max_width=20):
    """Apply frequency masking to spectrogram (works on single image)"""
    # Convert to numpy for easier manipulation
    spec = spectrogram.numpy()
    height, width, channels = spec.shape
    
    num_masks = tf.random.uniform([], 1, max_masks + 1, dtype=tf.int32).numpy()
    
    for _ in range(num_masks):
        f_width = tf.random.uniform([], 1, max_width + 1, dtype=tf.int32).numpy()
        f_start = tf.random.uniform([], 0, height - f_width, dtype=tf.int32).numpy()
        
        # Apply mask
        spec[f_start:f_start + f_width, :, :] = 0
    
    return tf.convert_to_tensor(spec, dtype=tf.float32)

def apply_time_masking(spectrogram, max_masks=2, max_width=50):
    """Apply time masking to spectrogram (works on single image)"""
    # Convert to numpy for easier manipulation
    spec = spectrogram.numpy()
    height, width, channels = spec.shape
    
    num_masks = tf.random.uniform([], 1, max_masks + 1, dtype=tf.int32).numpy()
    
    for _ in range(num_masks):
        t_width = tf.random.uniform([], 1, max_width + 1, dtype=tf.int32).numpy()
        t_start = tf.random.uniform([], 0, width - t_width, dtype=tf.int32).numpy()
        
        # Apply mask
        spec[:, t_start:t_start + t_width, :] = 0
    
    return tf.convert_to_tensor(spec, dtype=tf.float32)

def main():
    # Download and load the GTZAN dataset
    # dataset_path = download_gtzan_dataset()
    dataset_path = '/home/bill/code/AI/data/Data/genres_original'
    
    # Process the dataset and extract features with augmentation
    use_augmentation = True  # Set to True to enable data augmentation
    augmentation_factor = 1  # Number of augmented samples to create per original sample
    
    features, labels = preprocess_dataset(
        dataset_path, 
        apply_augmentation=use_augmentation,
        augmentation_factor=augmentation_factor
    )
    
    # Print some debug information about the shapes
    if features:
        print(f"Total features extracted: {len(features)}")
        if 'mel_spectrogram' in features[0]:
            shapes = [f['mel_spectrogram'].shape for f in features]
            unique_shapes = set(str(s) for s in shapes)
            print(f"Mel spectrogram shapes: {unique_shapes}")
    
    # Prepare data for the model
    X_train, X_test, y_train, y_test, label_encoder = prepare_data_for_model(features, labels)
    
    # Get the number of classes
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")
    
    # Build the CRNN model
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # Choose whether to use a custom model or a pre-trained one
    use_pretrained = True  # Set to True to use pre-trained model
    
    if use_pretrained:
        model = load_pretrained_model(input_shape, num_classes)
        print("Using pre-trained model with fine-tuning")
    else:
        model = build_crnn_model(input_shape, num_classes)
        print("Using custom CRNN model")
    
    # Find optimal learning rate
    optimal_lr = find_optimal_learning_rate(model, X_train, y_train)
    
    # Create learning rate schedule with warm-up
    initial_learning_rate = optimal_lr
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    # Create a warm-up learning rate schedule wrapper
    def warmup_cosine_decay(epoch):
        # Warm up for 3 epochs
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            # After warm-up, use cosine decay
            return lr_schedule(epoch - warmup_epochs)
    
    # Compile the model with the cosine decay learning rate schedule
    model.compile(
        optimizer=optimizers.Adam(learning_rate=warmup_cosine_decay(0)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Create more advanced data augmentation
    data_augmentation = create_advanced_data_augmentation()
    
    # Adjust batch size for more stable training
    batch_size = 32  # Increased from 15
    
    # Create LR scheduler callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(warmup_cosine_decay)
    
    # Train the model
    history = model.fit(
        tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x: data_augmentation(x, training=True)
        ).flow(X_train, y_train, batch_size=batch_size),
        epochs=100,  # Increased epochs
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),  # Increased patience
            lr_scheduler,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.path.dirname(__file__), "best_model.keras"),
                save_best_only=True,
                monitor="val_accuracy"
            )
        ]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model in .keras format
    model_save_path = os.path.join(os.path.dirname(__file__), "music_classification_model.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "training_history.png"))
    plt.show()

if __name__ == "__main__":
    main()