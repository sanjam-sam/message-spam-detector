# Message Spam Detector

A machine learning mini-project that classifies a short email/SMS subject as Spam or Not Spam using a trained Naive Bayes model and a Flask web app.

## Project Goal

This project demonstrates a complete end-to-end ML workflow:

1. Prepare labeled text data.
2. Convert text into numeric features.
3. Train and evaluate a classifier.
4. Save the trained model for reuse.
5. Serve predictions through a web interface.

## Why This Stack Was Used

### 1) Python
Python is widely used for machine learning and has strong libraries for both model training and deployment.

### 2) Pandas
Used to organize the training dataset into a tabular DataFrame format, which is easy to inspect and process.

### 3) TF-IDF Vectorizer
Used to convert text into numeric vectors for ML.

Why TF-IDF instead of basic counts:
- It reduces the impact of very common words.
- It gives more importance to words/phrases that are more informative for spam detection.
- It improves performance for short-text classification tasks.

### 4) Multinomial Naive Bayes
Used as the classification model.

Why this model:
- Fast to train.
- Works very well with sparse text vectors.
- A classic and strong baseline for spam filtering.

### 5) Train/Test Split with Stratify
Used to evaluate the model on unseen data.

Why stratify was used:
- Keeps the same spam/not-spam class ratio in both train and test sets.
- Prevents misleading evaluation when classes are imbalanced.

### 6) Confusion Matrix + Classification Metrics
Used to understand model behavior in detail.

What they tell us:
- Accuracy: overall correct predictions.
- Confusion matrix: exact counts of caught spam, missed spam, and wrongly blocked normal messages.
- Classification report: precision, recall, and F1 per class.

### 7) Joblib Model Persistence
Used to save the trained vectorizer and model as files:
- python-for-ai/vectorizer.pkl
- python-for-ai/spam_model.pkl

Why this matters:
- The app does not retrain on every request.
- Startup is faster.
- Deployment is simpler and reproducible.

### 8) Flask Backend
Used to expose an HTTP API and serve the web page.

Why Flask:
- Lightweight and beginner-friendly.
- Perfect for simple ML inference APIs.
- Easy to connect with HTML/JavaScript frontend.

### 9) HTML/CSS/JavaScript Frontend
Used for a simple interactive UI with:
- One input textbox for message subject.
- Check button for prediction.
- Clear button for reset.

Why this approach:
- No heavy frontend framework needed.
- Easy to understand and modify.

## Project Structure

```text
Python/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ paris_weather.csv
└─ python-for-ai/
   ├─ app.py
   ├─ spam_detector.py
   ├─ spam_detector.html
   ├─ vectorizer.pkl
   ├─ spam_model.pkl
   └─ ...other learning scripts
```

## How It Works Internally

1. spam_detector.py builds the dataset and trains the model.
2. TF-IDF transforms messages to vectors.
3. MultinomialNB learns the spam patterns.
4. The trained vectorizer and model are saved using Joblib.
5. app.py loads these saved files once on startup.
6. The HTML page sends user input to POST /predict.
7. Flask predicts and returns JSON with spam or ok.
8. The page displays Result: Spam or Result: Not Spam.

## Setup Instructions

## Prerequisites

- Python 3.11+
- pip

## Install dependencies

From project root (Python/):

```bash
pip install -r requirements.txt
```

## Train and save model files (run once or after data changes)

```bash
python python-for-ai/spam_detector.py
```

Expected output includes:
- Accuracy
- Confusion Matrix
- Saved model files message

## Start Flask app

```bash
python python-for-ai/app.py
```

Open in browser:

- http://127.0.0.1:5000

## API Contract

Endpoint:

- POST /predict

Request JSON:

```json
{
  "subject": "Win free cash now"
}
```

Response JSON:

```json
{
  "prediction": "spam"
}
```

Possible values:
- spam
- ok

## Current Limitations

- Training data is small and handcrafted.
- Predictions may not generalize to all real-world messages.
- No probability/confidence is shown yet.

## Recommended Next Improvements

1. Replace toy dataset with a larger real dataset (for example SMS Spam Collection).
2. Add probability score to response and UI.
3. Add automated tests for API and prediction behavior.
4. Add preprocessing (lowercasing, punctuation handling, optional stemming/lemmatization).
5. Add model versioning and experiment tracking.

## GitHub Upload Steps

If this folder is not already a git repository, run from project root (Python/):

```bash
git init
git add .
git commit -m "Add Message Spam Detector project"
git branch -M main
```

Create an empty GitHub repository, then connect and push:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Notes

- If push asks for authentication, use GitHub sign-in flow or a Personal Access Token.
- If you retrain the model, the .pkl files will update.
