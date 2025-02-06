import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Define paths
data_dir = "data"
normal_dir = os.path.join(data_dir, "normal")
anomalous_dir = os.path.join(data_dir, "anomalous")

# Load dataset
def load_audio_files(directory, label):
    features, labels = [], []
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            y, sr = librosa.load(os.path.join(directory, file), sr=16000)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            features.append(mfcc)
            labels.append(label)
    return np.array(features), np.array(labels)

# Prepare dataset
X_normal, y_normal = load_audio_files(normal_dir, 0)
X_anomalous, y_anomalous = load_audio_files(anomalous_dir, 1)

X = np.vstack((X_normal, X_anomalous))
y = np.hstack((y_normal, y_anomalous))

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model/fan_fault_detection.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training complete and saved as TensorFlow Lite model.")
