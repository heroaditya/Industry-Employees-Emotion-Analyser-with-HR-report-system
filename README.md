Employee well-being is crucial for organizational success, impacting productivity, job satisfaction, and retention. Traditional methods such as surveys and periodic reviews often fail to capture real-time emotional fluctuations, making timely intervention challenging.
EmpathicNet addresses this gap by leveraging AI-driven emotion recognition using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTML) networks for facial expression analysis. It integrates speech emotion recognition  through CNN-LSTM models and performs text sentiment analysis using fine-tuned DistilBERT. This multimodal approach provides support and intervention.
In addition to emotion detection, EmpathicNet offers personalized task recommendations based on emotional states, promoting well-being and productivity. A built-in HR alert system notifies relevant personnel of concerning emotional trends, enabling timely assistance.
Designed for real-time performance, EmpathicNet uses lightweight models optimized for CPU execution, ensuring accessibility without the need for high-end hardware. By providing actionable insights and fostering a supportive work environment, EmpathicNet contributes to a healthier, more productive workplace. This research exemplifies the application of AI in workplace well-being management, advancing the integration of empathetic technology into organizational settings.

I.	PROPOSED METHODOLOGY
a.	System Overview
The proposed system consists of the three main emotion detection models operating in parallel: Facial Emotion recognition (FER), Speech Emotion Recognition (SER), and Text-Based Emotion Analysis (TEA). Each model is optimized or its respective input type. The results from these models are combined using a decision-level fusion mechanism to provide a final emotion prediction. The system also includes a dashboard for real-time visualization of emotions, a task recommendation engine for suggesting suitable activities based on detected emotions, and an HR alert system to flag concerning emotional patterns.
The workflow of EmpaticNet is illustrated in Fig.1.
![image](https://github.com/user-attachments/assets/30c9a9c5-c0c1-4ba1-8258-6ebaa49500c9)
b.	Data Preprocessing
Preprocessing plays a critical role in ensuring data consistency and improving model accuracy. Each data modality undergoes a unique preprocessing pipeline.
![image](https://github.com/user-attachments/assets/125f8545-8dfc-41d9-b137-84850357dce3)
1.	Facial Data Preprocessing
Facial images are captured using a webcam or uploaded as image files. OpenCV is employed to extract video frames, while MediaPipe Face Mesh detects facial landmarks. The detected faces are cropped are resized to 224x224x3 pixels for compatability with the MobileNetV2 model. Data augmentation techniques such as rotation, flipping, and brightness adjustments are applied to increase model generalization.
2.	Speech Data Preprocessing
For speech emoyion recognition, audio is captured in real-time or uploaded in .wav format. Mel-Frequency Capstral Coefficientts (MFCCs), chroma features, and mel-spectrograms are extracted using Librosa. These features capture both spectral and temporal aspects of apeech, forming a 40xT matrix, where T represents the time steps. Speech augmentation, including noise addition and pitch shifting, is applied for robust model performance.
3.	Text Data Preprocessing
Texual data is tokenized using the WordPiece tokenizer and process3ed using DistilBERT. Stopwords and special characters are removed, and text sequences are padded to maintain uniform input size. The tokenized input is embedded using pre-trained DistilBERT to generate contexual embeddings for emotion classification.
c.	Model Architecture
EmpaticNet incorporates specialized deep learning models for each modality. The following describes the architectural details:
1.	Facial Emotion Recognition (FER)
Facial emotions are classified using a MobileNetV2 model du to its lightweight architecture and efficient feature extraction capabilities. The model consistes of multiple depthwise separable convolutional layers followed by batch normalization and ReLU activation. A global average pooling lyer is used to reduce the feature map size, and the final output is passed through a fully connected layer with softmax activation for classification into seven categories: Happy, Sad, Angry, Fearful, Disgusted, Surprised  and Neutral.
E_FER=Softmax(W.X+b).
Where:
E_FER = Facial emotion outpur
W = Weight matrix
X = Extracted feature vector
b = Bias term
2.	Speech Emotion Recognition (SER)
Speech emotions are detected using a CNN-LSTM hybrid model. The CNN layers extract spatial features, while the LSTM layers capture temporal dependencies in speech. The final output layer applies a softmax activation to classify the emotions into six categories: Calm, Happy, Fearful, Angry, Disgusted, and Sad.
The CNN component is defined as follows:
X_c=ReLU(Conv2D(X_s ))
Where:
X_c= CNN feature map
X_s= Speech feature matrix
The LSTM component then processes the CNN output:
E_SER=Softmax(LSTM(X_c).
3.	Text-Based Emotion Analysis (TEA)
The TEA module uses a fine-tuned DistilBERT model to analyze text sentiment. DistilBERT generates contexual embeddings, which are passed through dense layers for classification into seven categories: Joy, Sadness, Anger, Fear, Disgust, Surprise, and Neutral. Transfer learning is applied to ensure optimal 
performance.
d.	Multimodal Fusion
To combine the outputs from the three models, EmpaticNet uses a weighted decision-level fusion mechanism. The confidence scores from each model is used to calculate the final emotion prediction using a weighted sum approach:
E= w_1.E_FER+w_2.E_SER+w_3.E_TEA
Where:
E = Final emotion output
w_1,w_2,w_3 = Model-specific confidence-based weights
The weight assignment is adaptive, prioritizing the model with the highest confidence score. This ensures robustness and accuracy, particularly in scenarios where one modality may fail (e.g., poor lighting for facial recognition).
e.	HR alert System
EmpaticNet tracks employees’ emotional states over time using a time-series analysis. If negative emotions such as stress, sadness, or anger are detected for a defined period, the HR system receives an automatic alert. The alert includes an emotional trend report and recommendations for HR interventions, promoting early support and conflict resolution.
f.	Task Recommendation Engine
To enhance workplace productivity, the system suggests tasks aligned with the employee’s emotional state. Positive emotions lead to tasks requiring creativity and collaboration, while negative emotions result in recommendations for relaxation activities, mindfulness exercises, or light administrative tasks. Reinforcement learning is applied to improve task recommendations based on employee feedback.
g.	Evaluation Metrics
EmpaticNet is evaluated using multiple metrics to ensure accuracy and reliability. The primary evaluation metrics include:
Accuracy: Measures overall model correctness.
Precision and Recall: Evaluates model performance for each emotion class.
F1-Score: Balances precision and recall using the harmonic mean.
Confusion Metrix: Provides detailed insights into model misclassifications.
Accuracy=(TP+TN)/(TP+FP+TN+FN)
Where:
TP = True Positives, TN = True Negatives
FP = False Positives, FN = False Negatives
