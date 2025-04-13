import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.pipeline import Pipeline
import pickle  # Using pickle for saving sklearn models

# --- 1. Load and Prepare the Dataset ---
print("Loading the dataset...")
try:
    df = pd.read_csv('datasets/goemotions_1.csv')
except FileNotFoundError:
    print("Error: The file 'datasets/goemotions_1.csv' was not found.")
    exit()

# Identify emotion labels based on the head() output
emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
                  'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                  'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                  'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                  'pride', 'realization', 'relief', 'remorse', 'sadness',
                  'surprise', 'neutral']

# Separate features (text) and target labels (emotions)
X = df['text']
y = df[emotion_labels]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")
print(f"Number of emotion labels: {len(emotion_labels)}")

# --- 2. Define the Model and Pipeline for Multi-label Classification ---
print("\nDefining the model and pipeline for multi-label classification...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=42)))
])

# --- 3. Define Hyperparameter Grid for Tuning ---
print("\nDefining hyperparameter grid...")
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2, 3],
    'clf__estimator__C': [0.1, 1, 10],
    'clf__estimator__penalty': ['l1', 'l2']
}

# --- 4. Perform Hyperparameter Tuning using GridSearchCV ---
print("\nPerforming hyperparameter tuning...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# --- 5. Evaluate the Best Model ---
print("\nEvaluating the best model on the test set...")
y_pred = best_model.predict(X_test)

# Evaluate using accuracy (exact match of all labels)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (exact match): {accuracy:.4f}")

# Evaluate using Hamming Loss (average fraction of incorrectly predicted labels)
hamming = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hamming:.4f}")

print("\nPer-class Classification Reports:")
for i, label in enumerate(emotion_labels):
    print(f"\n--- Classification Report for {label} ---")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# --- 6. Save the Trained Model using pickle (for sklearn models) ---
print("\nSaving the trained model...")
model_filename = 'emotion_model_multilabel.pkl'  # Changed extension to .pkl
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"Trained multi-label emotion model saved as '{model_filename}'")

print("\n--- Training and Evaluation Complete ---")