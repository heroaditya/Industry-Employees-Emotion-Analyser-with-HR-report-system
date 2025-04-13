import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras_tuner import RandomSearch

# Load Data
df = pd.read_csv('datasets/fer2013.csv')
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1) / 255.0)

# Extract Features and Labels
X = np.stack(df['pixels'].values)
y = to_categorical(df['emotion'].values, num_classes=7)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('filters_1', 32, 128, step=32), (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('filters_2', 64, 128, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('filters_3', 64, 128, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('dense_units', 128, 512, step=64), activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter Tuning using Random Search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='facial_emotion'
)

# Perform Search
tuner.search(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters()[0]

# Build and Train Best Model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate Model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save Model
best_model.save('models/emotion_model_face.h5')

# Classification Report
y_pred = np.argmax(best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()
