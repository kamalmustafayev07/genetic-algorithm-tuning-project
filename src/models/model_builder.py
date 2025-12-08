import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def decode_vector(vector):
    return {
        'filters1': int(round(vector[0])),
        'kernel1': int(round(vector[1])),
        'filters2': int(round(vector[2])),
        'kernel2': int(round(vector[3])),
        'dropout_rate': float(vector[4]),
        'dense1': int(round(vector[5])),
        'dense2': int(round(vector[6])),
        'learning_rate': 10 ** vector[7]
    }

def build_model_from_vector(vector, input_shape=(28, 28, 1)):
    params = decode_vector(vector)

    model = Sequential([
        Conv2D(params['filters1'], (params['kernel1'], params['kernel1']),
               activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(params['filters2'], (params['kernel2'], params['kernel2']),
               activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(params['dropout_rate']),
        Flatten(),
        Dense(params['dense1'], activation='relu'),
        Dense(params['dense2'], activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, params
