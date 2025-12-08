# cnn_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Check for GPU availability and configure TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable GPU memory growth to avoid allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found; falling back to CPU.")

# Global evaluation counter (for logging, as in project examples)
eval_count = 0

# Load and prepare MNIST data once (outside function for efficiency)
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full = X_train_full.reshape(X_train_full.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Split train_full into train (54k) and val (6k)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# One-hot encode
y_train_cat = to_categorical(y_train, num_classes=10)
y_val_cat = to_categorical(y_val, num_classes=10)
# y_test_cat = to_categorical(y_test, 10)  # Use for final test eval, not optimization

def evaluate_solution(vector):
    global eval_count
    eval_count += 1
    
    try:
        # Decode vector (Section 6: rounding for ints, exp for LR)
        filters1 = int(round(vector[0]))   # x0
        kernel1 = int(round(vector[1]))    # x1
        filters2 = int(round(vector[2]))   # x2
        kernel2 = int(round(vector[3]))    # x3
        dropout_rate = vector[4]           # x4 (float)
        dense1 = int(round(vector[5]))     # x5
        dense2 = int(round(vector[6]))     # x6
        learning_rate = 10 ** vector[7]    # x7
        
        # Logging (as in project examples)
        print(f"\nEvaluation {eval_count}:")
        print(f"Params: filters1={filters1}, kernel1={kernel1}, filters2={filters2}, kernel2={kernel2}, "
              f"dropout={dropout_rate:.3f}, dense1={dense1}, dense2={dense2}, lr={learning_rate:.6f}")
        
        # Build model (Section 3 architecture)
        model = Sequential([
            Conv2D(filters=filters1, kernel_size=(kernel1, kernel1), activation='relu', padding='same', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=filters2, kernel_size=(kernel2, kernel2), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=dropout_rate),
            Flatten(),
            Dense(units=dense1, activation='relu'),
            Dense(units=dense2, activation='relu'),
            Dense(units=10, activation='softmax')
        ])
        
        # Compile
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train (will use GPU if available)
        history = model.fit(X_train, y_train_cat, epochs=5, batch_size=128, verbose=0,
                            validation_data=(X_val, y_val_cat))
        
        # Compute loss
        val_accuracy = max(history.history['val_accuracy'])
        loss = 1 - val_accuracy
        print(f"Validation accuracy: {val_accuracy:.4f} (loss: {loss:.4f})")
        
        # Clean up
        tf.keras.backend.clear_session()
        del model
        
        return loss
    
    except Exception as e:
        print(f"Error in evaluation {eval_count}: {str(e)}")
        return 1.0  # High loss on failure