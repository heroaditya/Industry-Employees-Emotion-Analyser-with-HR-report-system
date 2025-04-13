import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_tuner import RandomSearch

# Path to dataset
dataset_path = 'datasets/RAVDESS/'

# Extract features using MFCC
def extract_features(file_path, max_pad_len=200):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Ensure mfccs.shape[1] does not exceed max_pad_len
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]  # Trim to max_pad_len
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs


# Load Data
data = []
labels = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            label = file.split('-')[2]
            features = extract_features(file_path)
            data.append(features)
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)  # Convert labels to integer indices

num_classes = len(np.unique(y))  # Dynamically detect number of classes
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)  # One-hot encode labels

# Print to verify
print(f"Detected {num_classes} unique classes.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build Model using Keras Tuner
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('filters_1', 32, 128, step=32), (3, 3), activation='relu', input_shape=(40, 200, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('filters_2', 64, 128, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((128, -1)))
    model.add(layers.LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True))
    model.add(layers.LSTM(hp.Int('lstm_units_2', 32, 128, step=32)))
    model.add(layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))

    # **Ensure correct output shape**
    model.add(layers.Dense(num_classes, activation='softmax'))  # Use dynamic class count

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform Hyperparameter Tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='speech_emotion'
)

tuner.search(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters()[0]

# Train Best Model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate Model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save Model
best_model.save('models/emotion_model_speech.h5')

# Plot Results
plt.figure(figsize=(10, 5))
sns.heatmap(pd.DataFrame(history.history).corr(), annot=True, cmap='coolwarm')
plt.title('Training Correlation Heatmap')
plt.show()
