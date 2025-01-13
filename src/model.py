from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, LeakyReLU, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam

def crear_modelo():
    modelo = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(2048, activation='relu'),
        LeakyReLU(alpha=0.01),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.0005),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo
