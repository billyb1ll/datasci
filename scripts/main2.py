#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Reshape, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import _path  # noqa: F401

# In[1]:


import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  # Make sure this im


# In[2]:


DATASET_PATH = "data/Data/genres_original"
GENRES = ['blues', 'classical', 'jazz', 'pop', 'rock']
SAMPLE_RATE = 22050

for dirname, _, filenames in os.walk('data/Data/genres_original'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


def extract_features(file_path, sr=22050, duration=30):
    """
    สกัดคุณลักษณะ Mel-spectrogram จากไฟล์เสียง

    Parameters:
        file_path (str): เส้นทางไฟล์เสียง
        sr (int): อัตราการสุ่มตัวอย่าง
        duration (int): ระยะเวลาที่ใช้ (วินาที)

    Returns:
        mel_spectrogram (np.array): Mel-spectrogram
    """
    # Ensure numpy is imported at the function level if not available globally
    import numpy as np

    try:
        # Try the default loading method
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        try:
            # Try with audioread backend explicitly
            import warnings
            warnings.filterwarnings('ignore')
            y, sr = librosa.load(
                file_path, sr=sr, duration=duration, res_type='kaiser_fast')
        except Exception as e:
            print(f"Second attempt failed: {e}")
            try:
                # Try with pydub as last resort
                from pydub import AudioSegment

                audio = AudioSegment.from_file(file_path)
                samples = np.array(audio.get_array_of_samples())
                # Convert to float32 and normalize
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                    # Take average of both channels
                    samples = samples.mean(axis=1)
                y = samples.astype(np.float32) / np.iinfo(samples.dtype).max

                if sr != audio.frame_rate:
                    # Need to resample
                    import resampy
                    y = resampy.resample(y, audio.frame_rate, sr)
            except Exception as e:
                print(f"All loading methods failed for {file_path}: {e}")
                # Return empty array with correct shape
                y = np.zeros(sr * duration)

    # Ensure the audio is the right length
    if len(y) < sr * duration:
        # Pad if too short
        y = np.pad(y, (0, sr * duration - len(y)))
    elif len(y) > sr * duration:
        # Trim if too long
        y = y[:sr * duration]

    # สกัด Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    # แปลงเป็น Decibel scale
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram


# In[4]:


# การโหลดและจัดเตรียมชุดข้อมูล
def prepare_dataset(dataset_path, genres, min_samples_per_class=None):
    """
    เตรียมชุดข้อมูลสำหรับการฝึกฝน พร้อมตรวจสอบจำนวนตัวอย่างในแต่ละประเภท

    Parameters:
        dataset_path (str): เส้นทางไปยังโฟลเดอร์ที่มีไฟล์เสียง
        genres (list): รายการประเภทดนตรีที่ต้องการจำแนก
        min_samples_per_class (int): จำนวนตัวอย่างขั้นต่ำที่ต้องการสำหรับแต่ละประเภท

    Returns:
        X (np.array): อาร์เรย์ของคุณลักษณะ
        y (np.array): อาร์เรย์ของป้ายกำกับ
    """
    features = []
    labels = []
    samples_per_genre = {}

    for genre in genres:
        samples_per_genre[genre] = 0
        genre_path = os.path.join(dataset_path, genre)
        try:
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                    file_path = os.path.join(genre_path, file_name)
                    print(f'Extracting features from {file_path}')
                    mel_spec = extract_features(file_path)
                    features.append(mel_spec)
                    labels.append(genre)
                    samples_per_genre[genre] += 1

                    # หยุดเมื่อได้จำนวนตัวอย่างที่ต้องการ
                    if min_samples_per_class and samples_per_genre[genre] >= min_samples_per_class:
                        break
        except Exception as e:
            print(f"Error processing genre {genre}: {e}")

    # แสดงจำนวนตัวอย่างในแต่ละประเภท
    for genre, count in samples_per_genre.items():
        print(f"{genre}: {count} samples")

    # แปลงป้ายกำกับเป็นตัวเลข
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # แปลงลิสต์เป็น numpy array
    X = np.array(features)
    y = np.array(y)

    return X, y, label_encoder, samples_per_genre


# In[5]:


# สร้างชุดข้อมูลสำหรับการฝึกฝนและทดสอบ
def create_train_test_data(X, y, test_size=0.2, val_size=0.2):
    """
    แบ่งข้อมูลเป็นชุดฝึกฝน ตรวจสอบ และทดสอบ

    Parameters:
        X (np.array): อาร์เรย์ของคุณลักษณะ
        y (np.array): อาร์เรย์ของป้ายกำกับ
        test_size (float): สัดส่วนของชุดทดสอบ
        val_size (float): สัดส่วนของชุดตรวจสอบ

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # แบ่งข้อมูลออกเป็นชุดฝึกฝนและทดสอบก่อน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # จากชุดฝึกฝน แบ่งออกเป็นชุดฝึกฝนและตรวจสอบ
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
    )

    # ปรับรูปร่างข้อมูลให้เหมาะกับ CRNN (samples, time_steps, features, channels)
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    return X_train, X_val, X_test, y_train, y_val, y_test


# In[6]:


# ฟังก์ชันแสดงตัวอย่าง Mel-spectrogram
def plot_mel_spectrogram(mel_spectrogram, title='Mel Spectrogram'):
    """
    แสดง Mel-spectrogram ด้วย matplotlib

    Parameters:
        mel_spectrogram (np.array): Mel-spectrogram ที่ต้องการแสดง
        title (str): ชื่อกราฟ
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()


# Functions to check data quality and balance
def check_data_quality(X, threshold=-60):
    """
    ตรวจสอบคุณภาพของข้อมูล Mel-spectrogram

    Parameters:
        X (np.array): อาร์เรย์ของ Mel-spectrogram
        threshold (float): ค่า dB ต่ำสุดที่ยอมรับได้

    Returns:
        bad_indices (list): ดัชนีของตัวอย่างที่มีคุณภาพต่ำ
    """
    bad_indices = []
    for i, spec in enumerate(X):
        # ตรวจสอบว่า spectrogram มีค่าต่ำมากเกินไปหรือไม่
        if np.mean(spec) < threshold:
            bad_indices.append(i)
            print(
                f"Warning: Sample {i} may have low quality (mean dB: {np.mean(spec):.2f})")

    if bad_indices:
        print(
            f"Found {len(bad_indices)} potentially problematic samples out of {len(X)}")
    else:
        print("All samples passed quality check.")

    return bad_indices


def check_data_balance(y, label_encoder):
    """
    ตรวจสอบความสมดุลของข้อมูลในแต่ละคลาส

    Parameters:
        y (np.array): อาร์เรย์ของป้ายกำกับ
        label_encoder (LabelEncoder): เครื่องมือแปลงป้ายกำกับ

    Returns:
        class_counts (np.array): จำนวนตัวอย่างในแต่ละคลาส
        class_names (list): ชื่อของแต่ละคลาส
    """
    class_counts = np.bincount(y)
    class_names = label_encoder.classes_

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("Class distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"{name}: {count} samples")

    # ตรวจสอบความสมดุล
    if max(class_counts) / min(class_counts) > 1.5:
        print("\nWarning: Data imbalance detected. Consider resampling or using class weights.")

    return class_counts, class_names


# In[7]:


if __name__ == '__main__':
    dataset_path = '/home/bill/code/AI/data/Data/genres_original'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    # เตรียมข้อมูล
    X, y, label_encoder, samples_per_genre = prepare_dataset(
        dataset_path, genres)
    print(f"รูปร่างของคุณลักษณะ: {X.shape}")
    print(f"รูปร่างของป้ายกำกับ: {y.shape}")

    # ตรวจสอบคุณภาพข้อมูล
    bad_indices = check_data_quality(X, threshold=-60)

    # ตรวจสอบความสมดุลของข้อมูล
    class_counts, class_names = check_data_balance(y, label_encoder)

    # แสดงตัวอย่าง Mel-spectrogram
    plot_mel_spectrogram(
        X[0], title=f'Mel Spectrogram: {label_encoder.inverse_transform([y[0]])[0]}')

    # สร้างชุดข้อมูลฝึกฝน ตรวจสอบ และทดสอบ
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_data(
        X, y)
    print(f"รูปร่างของข้อมูลฝึกฝน: {X_train.shape}")
    print(f"รูปร่างของข้อมูลตรวจสอบ: {X_val.shape}")
    print(f"รูปร่างของข้อมูลทดสอบ: {X_test.shape}")


# In[8]:


def create_crnn_model(input_shape, num_classes, learning_rate=0.0005):
    """
    สร้างโมเดล CRNN สำหรับการจำแนกดนตรีพร้อมป้องกัน overfitting

    Parameters:
        input_shape (tuple): รูปร่างของอินพุต (time_steps, features, channels)
        num_classes (int): จำนวนประเภทดนตรี
        learning_rate (float): อัตราการเรียนรู้เริ่มต้น

    Returns:
        model (tf.keras.Model): โมเดล CRNN
    """
    # อินพุตเลเยอร์
    inputs = Input(shape=input_shape)

    # ส่วน CNN
    # ชั้นที่ 1
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.0015))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # ชั้นที่ 2
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # ชั้นที่ 3
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.0015))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.0015))(x)  # เพิ่ม Conv layer
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # ชั้นที่ 4
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.4)(x)

    # เปลี่ยนรูปร่างสำหรับ RNN
    _, height, width, channels = x.shape
    x = Reshape((width, height * channels))(x)

    # ส่วน RNN - จับความสัมพันธ์เชิงลำดับตามเวลา
    x = Bidirectional(LSTM(128, return_sequences=True,
                           recurrent_dropout=0.1,
                           recurrent_regularizer=tf.keras.regularizers.l2(0.002)))(x)
    x = Dropout(0.4)(x)

    # LSTM ชั้นที่ 2
    x = Bidirectional(LSTM(128, return_sequences=False,
                           recurrent_dropout=0.1,
                           recurrent_regularizer=tf.keras.regularizers.l2(0.002)))(x)
    x = Dropout(0.4)(x)

    # ชั้นเชื่อมต่อแบบเต็ม (Fully Connected Layer)
    x = Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = Dropout(0.5)(x)

    # เพิ่มอีกหนึ่งชั้น Dense แบบเล็กลง
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = Dropout(0.5)(x)

    # ชั้นเอาต์พุต
    outputs = Dense(num_classes, activation='softmax')(x)

    # สร้างโมเดล
    model = Model(inputs=inputs, outputs=outputs)

    # คอมไพล์โมเดล
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_crnn_model(model, X_train, y_train, X_val, y_val, X_test, batch_size=16, epochs=100,
                     model_path='crnn_music_genre_model', class_weights=None):
    """
    ฝึกฝนโมเดล CRNN พร้อมเทคนิคป้องกัน overfitting

    Parameters:
        model (tf.keras.Model): โมเดล CRNN
        X_train, y_train: ข้อมูลฝึกฝน
        X_val, y_val: ข้อมูลตรวจสอบ
        X_test: ข้อมูลทดสอบ
        batch_size (int): ขนาดของ batch
        epochs (int): จำนวนรอบการฝึกฝน
        model_path (str): เส้นทางสำหรับบันทึกโมเดล
        class_weights (dict): น้ำหนักสำหรับแต่ละคลาสในกรณีข้อมูลไม่สมดุล

    Returns:
        history: ประวัติการฝึกฝน
    """
    # เพิ่ม Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.15,  # เพิ่มการขยับภาพตามแนวแกน x
        height_shift_range=0.15,  # เพิ่มการขยับภาพตามแนวแกน y
        zoom_range=0.15,         # เพิ่มการย่อ/ขยายภาพ
        rotation_range=5,        # หมุนภาพเล็กน้อย
        brightness_range=[0.8, 1.2],  # ปรับความสว่าง
        fill_mode='constant',    # เติมขอบด้วยค่าคงที่
        horizontal_flip=False    # ไม่กลับด้านแนวนอน (time-dependent)
    )

    # กำหนด callbacks ที่ช่วยป้องกัน overfitting
    callbacks = [
        # หยุดก่อนกำหนดเมื่อ validation loss ไม่ลดลง
        EarlyStopping(
            monitor='val_loss',
            patience=25,         # เพิ่มความอดทน
            restore_best_weights=True,
            verbose=1
        ),
        # บันทึกโมเดลที่ดีที่สุด
        ModelCheckpoint(
            f'{model_path}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # ลดอัตราการเรียนรู้เมื่อ validation loss ไม่ลดลง
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.15,         # ลด learning rate ลง 85%
            patience=5,          # ลดความอดทนเพื่อปรับ LR เร็วขึ้น
            min_lr=1e-7,
            verbose=1
        ),
        # เพิ่ม TensorBoard callback สำหรับการวิเคราะห์
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_path}',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

    # Trim spectrograms to fixed width
    X_train = X_train[:, :, :259]  # ตัดความกว้างของ spectrogram
    X_val = X_val[:, :, :259]
    X_test = X_test[:, :, :259]

    # ฝึกฝนโมเดลด้วย data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,  # เพิ่มการถ่วงน้ำหนักคลาสหากข้อมูลไม่สมดุล
        verbose=1
    )

    # บันทึกโมเดลหลังการฝึกฝน
    model.save(f'{model_path}_final.keras')

    # บันทึก history เป็นไฟล์ numpy
    np.save(f'{model_path}_history.npy', history.history)

    print(f"โมเดลถูกบันทึกไว้ที่ '{model_path}_final.keras'")
    print(f"ประวัติการฝึกฝนถูกบันทึกไว้ที่ '{model_path}_history.npy'")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(f'{model_path}_training_history.png')
    plt.show()

    return history, X_train, X_val, X_test

# ฟังก์ชันใหม่สำหรับการประเมินโมเดลและแสดงผล


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    ประเมินประสิทธิภาพของโมเดลและแสดงผลการทำนาย

    Parameters:
        model (tf.keras.Model): โมเดลที่ฝึกฝนแล้ว
        X_test (np.array): ข้อมูลทดสอบ
        y_test (np.array): ป้ายกำกับที่ถูกต้องของข้อมูลทดสอบ
        label_encoder (LabelEncoder): เครื่องมือแปลงป้ายกำกับ
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # ทำนายคลาส
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # รายงานการจำแนก
    class_names = label_encoder.classes_
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    # สร้าง Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # แสดงตัวอย่างการทำนายผิด
    misclassified = np.where(y_pred_classes != y_test)[0]
    if len(misclassified) > 0:
        print(
            f"\nตัวอย่างการทำนายผิด ({min(5, len(misclassified))} ตัวอย่าง):")
        for i in range(min(5, len(misclassified))):
            idx = misclassified[i]
            true_label = label_encoder.inverse_transform([y_test[idx]])[0]
            pred_label = label_encoder.inverse_transform(
                [y_pred_classes[idx]])[0]
            confidence = np.max(y_pred[idx]) * 100
            print(
                f"ตัวอย่างที่ {idx}: ทำนายเป็น {pred_label} (ความมั่นใจ {confidence:.2f}%) แต่ที่จริงคือ {true_label}")


if __name__ == '__main__':
    dataset_path = '/home/bill/code/AI/data/Data/genres_original'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    # เตรียมข้อมูลพร้อมตรวจสอบจำนวนตัวอย่าง
    X, y, label_encoder, samples_per_genre = prepare_dataset(
        dataset_path, genres)
    print(f"รูปร่างของคุณลักษณะ: {X.shape}")
    print(f"รูปร่างของป้ายกำกับ: {y.shape}")

    # ตรวจสอบคุณภาพข้อมูล
    bad_indices = check_data_quality(X, threshold=-60)

    # ตรวจสอบความสมดุลของข้อมูล
    class_counts, class_names = check_data_balance(y, label_encoder)

    # สร้าง class weights เพื่อจัดการกับข้อมูลไม่สมดุล
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class weights:", class_weights_dict)

    # แสดงตัวอย่าง Mel-spectrogram
    plot_mel_spectrogram(
        X[0], title=f'Mel Spectrogram: {label_encoder.inverse_transform([y[0]])[0]}')

    # สร้างชุดข้อมูลฝึกฝน ตรวจสอบ และทดสอบ
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_data(
        X, y)
    print(f"รูปร่างของข้อมูลฝึกฝน: {X_train.shape}")
    print(f"รูปร่างของข้อมูลตรวจสอบ: {X_val.shape}")
    print(f"รูปร่างของข้อมูลทดสอบ: {X_test.shape}")

    # สร้าง balanced subsets สำหรับ validation และ test
    def oversample_minority_classes(X, y, target_class_count=None):
        from sklearn.utils import resample

        # หากไม่ระบุจำนวนตัวอย่างเป้าหมาย ให้ใช้ค่าสูงสุด
        if target_class_count is None:
            target_class_count = max(np.bincount(y))

        X_resampled = []
        y_resampled = []

        # สำหรับแต่ละคลาส
        for class_id in np.unique(y):
            # ดึงข้อมูลของคลาสนั้น
            X_class = X[y == class_id]
            y_class = y[y == class_id]

            # เพิ่มจำนวนตัวอย่าง
            if len(X_class) < target_class_count:
                X_class_resampled, y_class_resampled = resample(
                    X_class,
                    y_class,
                    replace=True,
                    n_samples=target_class_count,
                    random_state=42
                )
            else:
                X_class_resampled = X_class
                y_class_resampled = y_class

            X_resampled.extend(X_class_resampled)
            y_resampled.extend(y_class_resampled)

        return np.array(X_resampled), np.array(y_resampled)

    # ปรับความสมดุลของชุดข้อมูลฝึกฝน
    X_train_balanced, y_train_balanced = oversample_minority_classes(
        X_train, y_train)
    print(f"รูปร่างของข้อมูลฝึกฝนหลังปรับความสมดุล: {X_train_balanced.shape}")

    # สร้างและฝึกฝนโมเดล
    input_shape = (128, 259, 1)  # (freq_bins, time_frames, channels)
    num_classes = len(np.unique(y))

    # สร้างโมเดลใหม่ที่ปรับปรุงแล้ว
    model = create_crnn_model(input_shape, num_classes, learning_rate=0.0008)
    model.summary()

    # ฝึกฝนโมเดล
    history, X_train_processed, X_val_processed, X_test_processed = train_crnn_model(
        model,
        X_train_balanced, y_train_balanced,  # ใช้ข้อมูลที่ปรับความสมดุลแล้ว
        X_val, y_val,
        X_test,
        batch_size=16,  # ลดขนาด batch
        epochs=150,     # เพิ่มจำนวน epoch
        class_weights=class_weights_dict  # ใช้ class weights
    )

    # ประเมินโมเดล
    evaluate_model(model, X_test_processed, y_test, label_encoder)
