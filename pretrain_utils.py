import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import math
from tensorflow.keras.applications import (
    ResNet50V2, EfficientNetB0, MobileNetV3Large, 
    VGG16, Xception, InceptionV3
)

def convert_grayscale_to_rgb(grayscale_images):
    """
    Convert single-channel grayscale images to 3-channel RGB format
    for compatibility with pre-trained models
    
    Args:
        grayscale_images: Numpy array of shape (n, height, width, 1)
        
    Returns:
        rgb_images: Numpy array of shape (n, height, width, 3)
    """
    return np.repeat(grayscale_images, 3, axis=3)

def create_data_augmentation_layer():
    """
    Create a data augmentation pipeline for training
    
    Returns:
        data_augmentation: Sequential model with augmentation layers
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

def fine_tune_pretrained_model(model, num_layers_to_unfreeze=10):
    """
    Fine-tune the pre-trained model by unfreezing the last few layers
    
    Args:
        model: The pre-trained model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        
    Returns:
        model: The model with unfrozen layers
    """
    # Unfreeze the last n layers
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
        
    return model

def get_pretrained_model_for_spectrograms(input_shape, num_classes, model_name='resnet50v2'):
    """
    Load and configure a pre-trained model for audio spectrograms.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        model_name: Name of pre-trained model to use
        
    Returns:
        Compiled Keras model
    """
    # Ensure input has 3 channels for pre-trained models
    if input_shape[-1] != 3:
        raise ValueError(f"Pre-trained models require 3-channel input but got {input_shape}")
    
    # Dictionary of available pre-trained models
    model_dict = {
        'resnet50v2': {'model': ResNet50V2, 'preprocess': tf.keras.applications.resnet_v2.preprocess_input},
        'efficientnet': {'model': EfficientNetB0, 'preprocess': tf.keras.applications.efficientnet.preprocess_input},
        'mobilenetv3': {'model': MobileNetV3Large, 'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input},
        'vgg16': {'model': VGG16, 'preprocess': tf.keras.applications.vgg16.preprocess_input},
        'xception': {'model': Xception, 'preprocess': tf.keras.applications.xception.preprocess_input},
        'inception': {'model': InceptionV3, 'preprocess': tf.keras.applications.inception_v3.preprocess_input}
    }
    
    # Select model - default to ResNet50V2 if specified model not available
    if model_name.lower() not in model_dict:
        print(f"Warning: Model {model_name} not found. Using ResNet50V2 instead.")
        model_name = 'resnet50v2'
    
    print(f"Loading pre-trained model: {model_name}")
    
    # Create preprocessing layer
    preprocess_layer = model_dict[model_name.lower()]['preprocess']
    
    # Create base model
    base_model = model_dict[model_name.lower()]['model'](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    x = preprocess_layer(inputs)
    x = base_model(x)
    
    # Add pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    context_vector = layers.Dot(axes=1)([x, attention_weights])
    
    # Add classification head
    x = layers.Dense(512, activation='relu')(context_vector)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_lr_scheduler(scheduler_type='warmup_cosine_decay', initial_lr=0.001, epochs=30):
    """
    Create a learning rate scheduler.
    
    Args:
        scheduler_type: Type of scheduler to use
        initial_lr: Initial learning rate
        epochs: Total number of epochs
        
    Returns:
        Keras callback for learning rate scheduling
    """
    if scheduler_type == 'step_decay':
        def step_decay(epoch):
            drop_rate = 0.5
            epochs_drop = 5
            lr = initial_lr * math.pow(drop_rate, math.floor((1+epoch)/epochs_drop))
            return lr
        return tf.keras.callbacks.LearningRateScheduler(step_decay)
    
    elif scheduler_type == 'exponential_decay':
        def exponential_decay(epoch):
            decay_rate = 0.1
            return initial_lr * math.exp(-decay_rate*epoch)
        return tf.keras.callbacks.LearningRateScheduler(exponential_decay)
    
    elif scheduler_type == 'cosine_decay':
        def cosine_decay(epoch):
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            return initial_lr * cosine_decay
        return tf.keras.callbacks.LearningRateScheduler(cosine_decay)
    
    elif scheduler_type == 'one_cycle':
        def one_cycle(epoch):
            max_lr = initial_lr * 10
            if epoch < epochs // 2:
                return initial_lr + (max_lr - initial_lr) * (epoch / (epochs // 2))
            else:
                return max_lr - (max_lr - initial_lr / 10) * ((epoch - epochs // 2) / (epochs // 2))
        return tf.keras.callbacks.LearningRateScheduler(one_cycle)
    
    elif scheduler_type == 'warmup_cosine_decay':
        warmup_epochs = 5
        
        def warmup_cosine_decay(epoch):
            if epoch < warmup_epochs:
                return initial_lr * ((epoch + 1) / warmup_epochs)
            else:
                decay_epochs = epochs - warmup_epochs
                epoch_in_decay_range = epoch - warmup_epochs
                cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch_in_decay_range / decay_epochs))
                return initial_lr * cosine_decay
        return tf.keras.callbacks.LearningRateScheduler(warmup_cosine_decay)
    
    else:
        print(f"Unknown scheduler type: {scheduler_type}, using ReduceLROnPlateau")
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=initial_lr / 100,
            verbose=1
        )

def plot_lr_schedule(lr_scheduler, epochs=30, batches_per_epoch=None):
    """
    Plot the learning rate schedule.
    
    Args:
        lr_scheduler: Learning rate scheduler callback
        epochs: Number of epochs
        batches_per_epoch: Number of batches per epoch (for schedulers that update per batch)
    """
    # Extract the schedule function from the callback
    if hasattr(lr_scheduler, 'schedule'):
        lr_function = lr_scheduler.schedule
    else:
        print("Cannot plot learning rate: no schedule function found")
        return
    
    # Generate learning rates for each epoch
    learning_rates = [lr_function(epoch, 0) for epoch in range(epochs)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('/home/bill/code/AI/learning_rate_schedule.png')
    plt.show()

def unfreeze_model_layers(model, num_layers_to_unfreeze=10, verbose=True):
    """
    Unfreeze the last n layers of the base model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        verbose: Whether to print unfreezing information
    
    Returns:
        Model with unfrozen layers
    """
    # Find the base model
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            base_model = layer
            break
    else:
        if verbose:
            print("Could not identify base model - skipping layer unfreezing")
        return model
    
    # Unfreeze the last n layers
    if num_layers_to_unfreeze > 0:
        if verbose:
            print(f"Unfreezing last {num_layers_to_unfreeze} layers of base model for fine-tuning")
        
        base_model.trainable = True
        
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
    
    return model

def create_model_ensemble(input_shape, num_classes, model_names=['resnet50v2', 'efficientnet']):
    """
    Create an ensemble of pre-trained models.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        model_names: List of pre-trained models to use
        
    Returns:
        Ensemble model
    """
    models_list = []
    inputs = layers.Input(shape=input_shape)
    
    for model_name in model_names:
        # Create base model
        model = get_pretrained_model_for_spectrograms(input_shape, num_classes, model_name)
        # Get the output before the final dense layer
        for i in range(len(model.layers)-1, 0, -1):
            if isinstance(model.layers[i], layers.Dense) and model.layers[i].units == num_classes:
                dense_layer = model.layers[i]
                break
        
        # Get the input to the final dense layer
        feature_extractor = models.Model(
            inputs=model.input,
            outputs=model.layers[model.layers.index(dense_layer)-1].output
        )
        models_list.append(feature_extractor)
    
    # Combine features from all models
    features = [model(inputs) for model in models_list]
    
    if len(features) > 1:
        concatenated = layers.Concatenate()(features)
    else:
        concatenated = features[0]
    
    # Final classification layer
    x = layers.Dense(512, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    ensemble_model = models.Model(inputs=inputs, outputs=outputs)
    return ensemble_model
