import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

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
            print(f"Warning: Sample {i} may have low quality (mean dB: {np.mean(spec):.2f})")
    
    if bad_indices:
        print(f"Found {len(bad_indices)} potentially problematic samples out of {len(X)}")
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

def oversample_minority_classes(X, y, target_class_count=None):
    """
    เพิ่มจำนวนตัวอย่างในคลาสที่มีตัวอย่างน้อย
    
    Parameters:
        X (np.array): อาร์เรย์ของคุณลักษณะ
        y (np.array): อาร์เรย์ของป้ายกำกับ
        target_class_count (int): จำนวนตัวอย่างเป้าหมาย
        
    Returns:
        X_resampled (np.array): อาร์เรย์คุณลักษณะที่ปรับสมดุลแล้ว
        y_resampled (np.array): อาร์เรย์ป้ายกำกับที่ปรับสมดุลแล้ว
    """
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
