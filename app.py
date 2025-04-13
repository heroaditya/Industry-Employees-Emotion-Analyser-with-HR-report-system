import os
from flask import Flask, render_template, request, redirect, session, url_for, jsonify, flash
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from datetime import datetime
import librosa
import numpy as np
import soundfile as sf
import pickle
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import cv2
import joblib
from datetime import datetime
import csv
import base64
import seaborn as sns
import io

# ========== FLASK INIT ==========
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Email config (optional)
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='adityasharma96458@gmail.com',        # <-- replace
    MAIL_PASSWORD='Emp@123'            # <-- replace
)
mail = Mail(app)

# ========== PATH SETUP ==========
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ========== LOAD MODELS ==========
facial_model = load_model('models/emotion_model_face.h5')
speech_model = load_model('models/emotion_model_speech.h5')
with open('models/emotion_model_multilabel.pkl', 'rb') as f:
    text_model = pickle.load(f)

# ========== UTILS ==========
def save_user(email, password, role):
    filename = 'static/data/users.csv'
    os.makedirs('data', exist_ok=True)
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['email', 'password', 'role'])

    # Write new user
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([email, generate_password_hash(password), role])
def get_users():
    df = pd.read_csv('static/data/users.csv')
    return df.to_dict(orient='records')

def save_emotion(email, modality, emotion):
    df_path = 'static/data/emotion_logs.csv'
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=["auth_timestamp", "email", "modality", "emotion"])
    else:
        df = pd.read_csv(df_path)
    df = pd.concat([df, pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "email": email,
        "modality": modality,
        "emotion": emotion
    }])])
    df.to_csv(df_path, index=False)

def log_emotion(user_email, test_type, emotion):
    with open('static/data/emotion_logs.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_email, test_type, emotion])


# Labels from FER2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

speech_labels = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

text_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

def extract_features(path, n_mfcc=40, max_pad_len=200):
    try:
        audio, sample_rate = librosa.load(path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Padding/truncating to max_pad_len
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc  # Shape will be (40, 200)
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None
    

def send_alert_email(to, subject, body):
    msg = Message(subject, sender='your_email@gmail.com', recipients=[to])
    msg.body = body
    mail.send(msg)

# ========== ROUTES ==========
@app.route('/')
def home():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if not email or not password or not role:
            flash("All fields are required!", "danger")
            return render_template('register.html')

        save_user(email, password, role)
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = get_users()
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        role = request.form.get('role', '')
        for user in users:
            if user['email'] == email and check_password_hash(user['password_hash'], password):
                session['email'] = email
                session['role'] = user['role']
                return redirect('/dashboard') if user['role'] == 'employee' else redirect('/hr_dashboard')
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'role' not in session or session['role'] != 'employee':
        return redirect('/login')
    return render_template('employee_dashboard.html', email=session['email'])

@app.route('/logout')
def logout():
    session.clear()  # or del session['email'], etc.
    return redirect('/login')

@app.route('/hr_dashboard')
def hr_dashboard():
    return render_template('hr_dashboard.html')

@app.route("/get_emotion_data")
def get_emotion_data():
    data = []
    with open("static/data/emotion_logs.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                data.append({
                    "auth_time": row[0],
                    "email": row[1],
                    "modality": row[2],
                    "emotion": row[3],
                })
    return jsonify(data)


@app.route('/predict_facial', methods=['POST'])
def predict_facial():
    if 'image' not in request.files:
        return render_template('employee_dashboard.html', prediction="No image uploaded", test_type="Facial")

    file = request.files['image']
    if file.filename == '':
        return render_template('employee_dashboard.html', prediction="Empty file uploaded", test_type="Facial")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))

    predictions = facial_model.predict(img)
    predicted_label = emotion_labels[np.argmax(predictions)]

    # Log emotion (CSV)
    log_emotion(session.get('email', 'Unknown'), "Facial", predicted_label)

    # Chart
    plt.figure(figsize=(8, 4))
    plt.bar(emotion_labels, predictions[0])
    plt.xlabel("Emotions")
    plt.ylabel("Probability")
    plt.title(f"Facial Emotion Prediction: {predicted_label}")
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'], 'latest_chart.png'))
    plt.close()

    return render_template('employee_dashboard.html', prediction=predicted_label, test_type="Facial")

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    if 'audio' not in request.files:
        return render_template('employee_dashboard.html', prediction="No audio uploaded", test_type="Speech")

    file = request.files['audio']
    if file.filename == '':
        return render_template('employee_dashboard.html', prediction="Empty file uploaded", test_type="Speech")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    features = extract_features(filepath)
    if features is None:
        return render_template('employee_dashboard.html', prediction="Feature extraction failed", test_type="Speech")

    # Reshape: (40, 200) → (1, 40, 200, 1)
    features = np.expand_dims(features, axis=0)       # (1, 40, 200)
    features = np.expand_dims(features, axis=-1)      # (1, 40, 200, 1)

    prediction = speech_model.predict(features)
    predicted_label = speech_labels[np.argmax(prediction)]

    # Log emotion (CSV)
    log_emotion(session.get('email', 'Unknown'), "Speech", predicted_label)

    # Chart
    plt.figure(figsize=(8, 4))
    plt.bar(speech_labels, prediction[0])
    plt.xlabel("Emotions")
    plt.ylabel("Probability")
    plt.title(f"Speech Emotion Prediction: {predicted_label}")
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'], 'latest_chart.png'))
    plt.close()

    return render_template('employee_dashboard.html', prediction=predicted_label, test_type="Speech")

@app.route('/predict_text', methods=['POST'])
def predict_text():
    if 'role' not in session or session['role'] != 'employee':
        return redirect('/login')

    text = request.form.get('text')
    if not text:
        return render_template('employee_dashboard.html', prediction="No text entered", test_type="Text")

    try:
        # Predict probabilities
        prediction_proba = text_model.predict_proba([text])
        threshold = 0.5  # You can adjust based on performance

        # Extract positive class probabilities
        predicted_indices = [i for i, prob in enumerate(prediction_proba) if prob[0][1] >= threshold]

        if not predicted_indices:
            predicted_label = "Neutral"
        else:
            predicted_label = ', '.join([text_labels[i] for i in predicted_indices])

        # Save chart
        plt.figure(figsize=(10, 5))
        scores = [prob[0][1] for prob in prediction_proba]  # class 1 probs
        plt.bar(text_labels, scores)
        plt.xticks(rotation=90)
        plt.xlabel("Emotions")
        plt.ylabel("Probability")
        plt.title(f"Text Emotion Prediction: {predicted_label}")
        plt.tight_layout()
        plt.savefig(os.path.join(app.config['RESULT_FOLDER'], 'latest_chart.png'))
        plt.close()

        # Log emotion
        save_emotion(session['email'], 'Text', predicted_label)

        return render_template('employee_dashboard.html', prediction=predicted_label, test_type="Text")

    except Exception as e:
        print("Text prediction error:", e)
        return render_template('employee_dashboard.html', prediction="Prediction failed", test_type="Text")

if __name__ == '__main__':
    app.run(debug=True)
