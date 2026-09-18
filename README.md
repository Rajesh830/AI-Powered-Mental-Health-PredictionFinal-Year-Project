# AI-Powered Mental Health Prediction 

## Overview

This project is an **AI-powered mental health prediction ** designed to assess users' mental health status using a combination of **machine learning, deep learning, sentiment analysis, emotion detection, and Generative AI**.

The system provides an interactive **Streamlit-based web application** where users can complete mental health assessments such as **PHQ-9 and GAD-7**, provide information related to stress, sleep, activity, and social interactions, and interact with an AI-powered mental health assistant.

A **K-Nearest Neighbors (KNN)** machine learning model is used to predict the user's mental health status based on the collected features. In addition, a custom deep learning emotion recognition model analyzes facial expressions, while **VADER sentiment analysis** is used to analyze text-based sentiment.

The application also supports **Google Gemini API** integration to provide conversational AI responses and personalized supportive guidance.

> **Disclaimer:** This system is intended for educational and research purposes. It does not provide a medical diagnosis and should not replace assessment or treatment by a qualified mental-health professional.

## Features

* **Mental Health Prediction:** Predicts a user's mental health status using a trained KNN machine learning pipeline.
* **PHQ-9 Assessment:** Collects responses to nine questions related to depressive symptoms and calculates a PHQ-9 score.
* **GAD-7 Assessment:** Collects responses to seven questions related to anxiety symptoms and calculates a GAD-7 score.
* **Emotion Detection:** Uses a custom deep learning model to recognize facial emotions.
* **Facial Expression Analysis:** Uses OpenCV and Haar Cascade face detection to identify faces from camera input.
* **Sentiment Analysis:** Uses VADER to analyze the sentiment of user messages.
* **Text-Based AI Assistant:** Provides conversational support through the Streamlit interface.
* **Generative AI Integration:** Uses Google Gemini API for AI-generated responses when configured.
* **User Information:** Considers factors such as stress, sleep quality, activity level, social interactions, emotions, and sentiment.
* **MongoDB Integration:** Stores user information, chat history, PHQ-9 scores, and session summaries.
* **Secure Authentication:** Uses password hashing with bcrypt for user authentication.
* **Interactive Dashboard:** Provides a web-based interface for assessments, prediction, emotion analysis, and conversations.

## Technologies Used

* **Python**
* **Streamlit**
* **Scikit-learn**
* **K-Nearest Neighbors (KNN)**
* **TensorFlow / Keras**
* **OpenCV**
* **VADER Sentiment Analysis**
* **Google Gemini API**
* **MongoDB**
* **Pandas**
* **NumPy**
* **Joblib**
* **Pillow**
* **python-dotenv**
* **bcrypt**

## Machine Learning Model

The project uses a **K-Nearest Neighbors (KNN) classifier** for mental health status prediction.

The trained model is stored as:

```text
sklearn_knn_mental_health_pipeline.joblib
```

The model uses a combination of numerical and categorical features.

### Numerical Features

```text
final_phq9_score
final_gad7_score
stress_level
social_interactions
emotion_volatility_estimate
avg_sentiment_last_5
```

### Categorical Features

```text
sleep_quality
activity_level
final_emotion
final_text_sentiment
```

These features are preprocessed before being provided to the KNN classifier.

## Deep Learning Emotion Detection

The system includes a trained TensorFlow/Keras emotion recognition model:

```text
emotion_model.h5
```

The model recognizes the following facial emotions:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

OpenCV is used to detect faces from camera input, after which the detected facial region can be processed by the emotion recognition model.

## Mental Health Assessment

### PHQ-9

The application includes a **Patient Health Questionnaire-9 (PHQ-9)** assessment containing nine questions related to depressive symptoms.

Each response is converted into a numerical score:

```text
Not at all          → 0
Several days        → 1
More than half days → 2
Nearly every day    → 3
```

The responses are combined to calculate the final PHQ-9 score.

### GAD-7

The application also includes a **Generalized Anxiety Disorder-7 (GAD-7)** assessment containing seven questions related to anxiety symptoms.

The same response scoring system is used:

```text
Not at all          → 0
Several days        → 1
More than half days → 2
Nearly every day    → 3
```

The responses are used to calculate the final GAD-7 score.

## System Architecture

```text
                    USER
                      │
                      ▼
             Streamlit Web Interface
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
     PHQ-9          GAD-7       Chat Input
   Assessment      Assessment        │
        │             │               ▼
        │             │        Sentiment Analysis
        │             │          using VADER
        │             │               │
        └─────────────┼───────────────┘
                      │
                      ▼
              Feature Collection
                      │
        ┌─────────────┼──────────────┐
        │             │              │
        ▼             ▼              ▼
     Emotion       Stress/Sleep   Social Activity
    Detection       Factors          Factors
        │             │              │
        └─────────────┼──────────────┘
                      │
                      ▼
             Feature Preprocessing
                      │
                      ▼
             KNN Prediction Model
                      │
                      ▼
           Mental Health Prediction
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     AI Assistant          Supportive Guidance
          │
          ▼
       Gemini API
          │
          ▼
    Conversational Response

                      │
                      ▼
                   MongoDB
```

## Project Workflow

### Step 1: User Registration and Login

Users can create an account and log in to the application.

Passwords are protected using **bcrypt hashing** before being stored.

### Step 2: Mental Health Assessment

The user completes the PHQ-9 and GAD-7 questionnaires.

The system calculates:

```text
PHQ-9 Score
GAD-7 Score
```

These scores become important features for the machine learning prediction.

### Step 3: Lifestyle and Behavioral Information

The system collects additional information such as:

* Stress level
* Sleep quality
* Activity level
* Social interactions

These features provide additional context for the prediction model.

### Step 4: Emotion Detection

The application can analyze facial expressions using:

```text
OpenCV
     ↓
Haar Cascade Face Detection
     ↓
Facial Region
     ↓
TensorFlow/Keras Emotion Model
     ↓
Detected Emotion
```

The detected emotion can be incorporated into the overall feature set.

### Step 5: Text Sentiment Analysis

User messages are analyzed using **VADER Sentiment Analysis**.

The system derives sentiment-related information from conversations and maintains recent sentiment information for prediction.

### Step 6: Feature Engineering

The system combines the collected information into a feature vector containing:

```text
PHQ-9 Score
GAD-7 Score
Stress Level
Social Interactions
Emotion Volatility
Average Recent Sentiment
Sleep Quality
Activity Level
Final Emotion
Final Text Sentiment
```

### Step 7: Machine Learning Prediction

The processed feature vector is passed to the trained KNN pipeline:

```text
User Features
      ↓
Preprocessing
      ↓
Scaling / Encoding
      ↓
KNN Classifier
      ↓
Mental Health Status
```

### Step 8: AI-Based Support

The application can use the Google Gemini API to provide conversational responses and supportive information based on the user's interaction.

### Step 9: Data Storage

MongoDB is used to maintain application data such as:

```text
Users
Chat History
PHQ-9 Scores
Session Summaries
```

## Dataset

The project includes the dataset:

```text
oversampled_mental_health_dataset.csv
```

The dataset is used to train the machine learning model for mental health status prediction.

The training process includes:

* Data loading
* Data cleaning
* Feature selection
* Numerical feature preprocessing
* Categorical feature preprocessing
* Train/test splitting
* KNN model training
* Model evaluation
* Model serialization

## Training the Machine Learning Model

The training script is:

```text
train.py
```

Run:

```bash
python train.py
```

The script trains the KNN model and saves the resulting pipeline as:

```text
sklearn_knn_mental_health_pipeline.joblib
```

The project also contains:

```text
train1.py
```

which can be used as an additional training implementation.

## Project Structure

```text
FINAL CODE/
│
├── chat.py
│
├── train.py
├── train1.py
├── facede.py
│
├── emotion_model.h5
│
├── sklearn_knn_mental_health_pipeline.joblib
│
├── oversampled_mental_health_dataset.csv
│
├── .env
│
└── tfenv/
```

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd "FINAL CODE"
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install streamlit tensorflow opencv-python numpy pandas pillow python-dotenv pymongo bcrypt vaderSentiment scikit-learn joblib google-generativeai
```

Alternatively, create a `requirements.txt` file containing the project dependencies and run:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root directory.

Example:

```env
MONGO_URI=mongodb://localhost:27017
GOOGLE_API_KEY=your_google_api_key_here
```

The Google API key is used for Gemini-based AI responses.

**Do not upload your `.env` file or API key to GitHub.**

Add `.env` to `.gitignore`:

```text
.env
venv/
tfenv/
__pycache__/
```

## MongoDB Configuration

The application uses MongoDB for data storage.

The default MongoDB connection is:

```text
mongodb://localhost:27017
```

The application creates/uses the following database:

```text
mental_health_bot
```

Collections include:

```text
users
chats
phq_scores
session_summaries
```

Make sure MongoDB is running before starting the application.

## Running the Application

After activating the virtual environment, run:

```bash
streamlit run chat.py
```

The application will normally open at:

```text
http://localhost:8501
```

## Usage

1. Start the Streamlit application.
2. Register a user account or log in.
3. Complete the PHQ-9 assessment.
4. Complete the GAD-7 assessment.
5. Provide relevant lifestyle and behavioral information.
6. Interact with the mental health assistant.
7. Allow emotion detection when using the facial-expression feature.
8. Enter messages for sentiment analysis.
9. The system combines the collected features.
10. The KNN model generates a mental health status prediction.
11. The AI assistant provides supportive information.
12. Relevant session information can be stored in MongoDB.

## Example Input

```text
PHQ-9 Score: 12
GAD-7 Score: 9
Stress Level: Moderate
Sleep Quality: Poor
Activity Level: Low
Social Interactions: Low
Detected Emotion: Sad
Text Sentiment: Negative
```

## Example Prediction

```text
Mental Health Prediction:

The machine learning model has identified a
potential mental-health status category based
on the provided assessment and behavioral features.

Please consider discussing persistent or concerning
symptoms with a qualified mental-health professional.
```

## Example AI Assistant Interaction

```text
User:

I have been feeling stressed and unable to concentrate
on my studies recently.

AI Assistant:

It sounds like you may be experiencing a period of
increased stress. Consider taking regular breaks,
maintaining a consistent sleep schedule, staying
physically active, and talking with someone you trust.

If these difficulties continue or significantly affect
your daily life, consider speaking with a qualified
mental-health professional.
```

## Advantages

* Combines multiple AI and machine learning techniques.
* Uses both questionnaire-based and behavioral features.
* Supports facial emotion recognition.
* Includes sentiment analysis of user conversations.
* Provides machine-learning-based mental health prediction.
* Offers an interactive Streamlit interface.
* Supports persistent data storage through MongoDB.
* Can integrate Generative AI for conversational support.
* Combines structured assessments with AI-based interaction.

## Limitations

* The prediction is not a clinical diagnosis.
* Machine learning predictions depend on the quality and representativeness of the training dataset.
* Facial emotion recognition can be affected by lighting, camera quality, facial orientation, and other factors.
* Sentiment analysis cannot reliably determine a person's complete mental-health condition.
* AI-generated responses may occasionally be inaccurate or inappropriate.
* Professional clinical assessment is required for diagnosis and treatment decisions.

## Future Enhancements

Future versions of the project can include:

* Integration with additional validated mental-health assessment tools.
* More diverse and clinically validated datasets.
* Advanced deep learning models for prediction.
* Transformer-based sentiment and emotion analysis.
* Improved facial emotion recognition.
* Time-series analysis of mental-health trends.
* Personalized wellness dashboards.
* Early-warning notification systems.
* Integration with wearable-device data.
* Explainable AI for prediction results.
* Doctor/counselor verification workflows.
* Secure cloud deployment.
* Improved privacy and encryption.
* Multilingual mental-health support.
* Voice-based emotion and sentiment analysis.

## Applications

The proposed system can be explored for:

* Academic mental-health research
* Student wellness applications
* Mental-health awareness platforms
* Personal wellness monitoring
* Conversational mental-health assistants
* AI-assisted screening research
* Emotion and sentiment analysis research

## Conclusion

The **AI-Powered Mental Health Prediction and Support System** demonstrates how multiple artificial intelligence techniques can be combined to develop an interactive mental-health support platform.

The system integrates **PHQ-9 and GAD-7 assessments, KNN-based machine learning, facial emotion recognition using TensorFlow and OpenCV, VADER sentiment analysis, MongoDB, and Google Gemini Generative AI**.

By combining questionnaire responses, behavioral information, emotional signals, and textual sentiment, the project demonstrates a multimodal approach to mental-health prediction and AI-assisted support.





**AI-Powered Mental Health Prediction and Support System Using Machine Learning, Deep Learning, Sentiment Analysis and Generative AI**
