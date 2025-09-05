import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, class_names=None, save_dir=None):
    """
    Comprehensive evaluation of a trained model
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: One-hot encoded test labels
        class_names: List of class names (strings)
        save_dir: Directory to save evaluation plots
    """
    # Convert one-hot encoded labels back to class indices
    y_true = np.argmax(y_test, axis=1)
    
    # Get model predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print classification report
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names)
    else:
        report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=class_names if class_names else None)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/confusion_matrix.png")
    
    plt.show()
    
    # Analyze misclassifications
    misclassified_indices = np.where(y_pred != y_true)[0]
    print(f"\nNumber of misclassified samples: {len(misclassified_indices)}")
    
    if len(misclassified_indices) > 0:
        # Create a table showing the most confident misclassifications
        misclass_confidence = y_pred_proba[misclassified_indices, y_pred[misclassified_indices]]
        sorted_indices = np.argsort(misclass_confidence)[::-1]  # Sort by confidence (descending)
        
        # Show top 10 most confident misclassifications
        top_n = min(10, len(misclassified_indices))
        print("\nTop most confident misclassifications:")
        print("Index\tTrue\tPredicted\tConfidence")
        
        for i in range(top_n):
            idx = misclassified_indices[sorted_indices[i]]
            true_class = y_true[idx]
            pred_class = y_pred[idx]
            confidence = y_pred_proba[idx, pred_class]
            
            true_name = class_names[true_class] if class_names else str(true_class)
            pred_name = class_names[pred_class] if class_names else str(pred_class)
            
            print(f"{idx}\t{true_name}\t{pred_name}\t{confidence:.4f}")
    
    # Calculate per-class accuracy
    per_class_acc = np.zeros(len(np.unique(y_true)))
    for i in range(len(per_class_acc)):
        indices = np.where(y_true == i)[0]
        if len(indices) > 0:
            per_class_acc[i] = np.mean(y_pred[indices] == i)
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(per_class_acc)), per_class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    
    if class_names:
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # Set y-axis limit to allow space for text
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/per_class_accuracy.png")
    
    plt.show()
    
    return {
        'accuracy': np.mean(y_pred == y_true),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_training_history(history, save_path=None):
    """
    Plot training history with better visualization
    
    Args:
        history: History object returned by model.fit()
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    
    # Find best accuracy and epoch
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_acc = history.history['val_accuracy'][best_epoch]
    
    # Highlight best accuracy
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5)
    plt.scatter(best_epoch, best_acc, s=100, c='red', label=f'Best: {best_acc:.4f} (epoch {best_epoch+1})')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    
    # Find best loss and epoch
    best_epoch_loss = np.argmin(history.history['val_loss'])
    best_loss = history.history['val_loss'][best_epoch_loss]
    
    # Highlight best loss
    plt.axvline(x=best_epoch_loss, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.5)
    plt.scatter(best_epoch_loss, best_loss, s=100, c='red', label=f'Best: {best_loss:.4f} (epoch {best_epoch_loss+1})')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    print("This is a utility module. Import and use in your training script.")
