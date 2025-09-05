import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def find_learning_rate(model, train_dataset, min_lr=1e-7, max_lr=1e-1, steps=100, epochs=1, batch_size=32):
    """
    Learning rate finder implementation for TensorFlow/Keras models.
    
    Args:
        model: A compiled Keras model
        train_dataset: Tuple of (X_train, y_train) or a tf.data.Dataset
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        steps: Number of steps between min and max lr
        epochs: Number of epochs to train each mini-batch
        batch_size: Batch size for training
        
    Returns:
        tuple: (learning_rates, losses)
    """
    # Save initial model weights
    initial_weights = model.get_weights()
    
    # Create learning rate range with exponential scale
    lr_range = np.geomspace(min_lr, max_lr, steps)
    losses = []
    
    if isinstance(train_dataset, tuple):
        X_train, y_train = train_dataset
        dataset_size = len(X_train)
        
        # Create mini-batches
        for i, lr in enumerate(lr_range):
            print(f"Testing learning rate: {lr:.8f} [{i+1}/{len(lr_range)}]", end="\r")
            
            # Reset weights to initial state
            model.set_weights(initial_weights)
            
            # Set learning rate
            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
            
            # Select random batch
            indices = np.random.randint(0, dataset_size, batch_size)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Train for one batch
            history = model.fit(X_batch, y_batch, epochs=epochs, verbose=0)
            losses.append(history.history['loss'][0])
    else:
        # Assume train_dataset is a tf.data.Dataset
        for i, lr in enumerate(lr_range):
            print(f"Testing learning rate: {lr:.8f} [{i+1}/{len(lr_range)}]", end="\r")
            
            # Reset weights to initial state
            model.set_weights(initial_weights)
            
            # Set learning rate
            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
            
            # Get a batch
            for x_batch, y_batch in train_dataset.take(1):
                history = model.fit(x_batch, y_batch, epochs=epochs, verbose=0)
                losses.append(history.history['loss'][0])
                break
    
    # Restore initial weights
    model.set_weights(initial_weights)
    
    # Remove extreme values for better visualization
    smooth_losses = []
    for i in range(len(losses)):
        if i == 0:
            smooth_losses.append(losses[i])
        else:
            smooth_losses.append(0.1 * losses[i] + 0.9 * smooth_losses[i-1])
    
    # Find optimal learning rate (steepest descent)
    gradients = np.gradient(smooth_losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lr_range[optimal_idx]
    
    print(f"\nOptimal learning rate: {optimal_lr:.8f}")
    
    return lr_range, losses, smooth_losses, optimal_lr

def plot_learning_rate_finder(lr_range, losses, smooth_losses=None, optimal_lr=None, save_path=None):
    """Plot learning rate finder results"""
    plt.figure(figsize=(12, 6))
    plt.plot(lr_range, losses, 'b-', alpha=0.5, label='Raw loss')
    
    if smooth_losses is not None:
        plt.plot(lr_range, smooth_losses, 'r-', linewidth=2, label='Smoothed loss')
    
    if optimal_lr is not None:
        plt.axvline(x=optimal_lr, color='green', linestyle='--', label=f'Optimal LR: {optimal_lr:.8f}')
        # Suggest a slightly lower learning rate for training
        suggested_lr = optimal_lr / 10
        plt.axvline(x=suggested_lr, color='purple', linestyle=':', label=f'Suggested LR: {suggested_lr:.8f}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # 1. Create and compile your model
    # 2. Run:
    # lr_range, losses, smooth_losses, optimal_lr = find_learning_rate(model, (X_train, y_train))
    # 3. Plot:
    # plot_learning_rate_finder(lr_range, losses, smooth_losses, optimal_lr, "lr_finder_plot.png")
    print("This is a utility module. Import and use in your training script.")
