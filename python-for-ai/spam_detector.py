import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = {
    "message": [
        # spam
        "Win money now!", "Congratulations! You've won a prize!",
        "Earn cash quickly!", "Click here to claim your free gift!",
        "You have been selected for a cash reward!", "Buy cheap meds online now!",
        "Limited offer! Get rich fast!", "Free entry to win $1000!",
        "Call now to claim your lottery prize!", "Exclusive deal just for you!",
        "You won a free iPhone! Click to claim.", "Urgent: Your account has been compromised, verify now!",
        "Get paid to work from home!", "Double your income in 30 days!",
        "WINNER!! Claim your prize before it expires!", "Lowest mortgage rates, apply now!",
        "You are pre-approved for a $5000 loan!", "Act now! Limited time offer!",
        "Lose weight fast with this one weird trick!", "Make $500 a day from home!",
        # ok
        "Hello, how are you?", "Let's go for lunch.",
        "Meeting tomorrow.", "Can you send me the report?",
        "Are you coming to the party?", "I'll call you later.",
        "Please review the attached document.", "What time works for you?",
        "See you at 5pm.", "Don't forget to pick up groceries.",
        "Can we reschedule our meeting?", "Happy birthday! Hope you have a great day.",
        "The project deadline is next Friday.", "Did you watch the game last night?",
        "I left my keys at your place.", "Let me know when you're free to talk.",
        "Thanks for your help yesterday.", "I'll send you the files tonight.",
        "Are you free this weekend?", "Good morning, hope your day goes well!"
    ],
    "label": [
        "spam","spam","spam","spam","spam","spam","spam","spam","spam","spam",
        "spam","spam","spam","spam","spam","spam","spam","spam","spam","spam",
        "ok","ok","ok","ok","ok","ok","ok","ok","ok","ok",
        "ok","ok","ok","ok","ok","ok","ok","ok","ok","ok"
    ]
}
df = pd.DataFrame(data)

# TF-IDF weighs rare/important words higher than common ones
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# stratify ensures equal spam/ok ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

base_dir = Path(__file__).resolve().parent
joblib.dump(vectorizer, base_dir / "vectorizer.pkl")
joblib.dump(model, base_dir / "spam_model.pkl")

print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=["spam", "ok"])
cm_df = pd.DataFrame(cm,
    index=["Actual Spam", "Actual OK"],
    columns=["Predicted Spam", "Predicted OK"]
)
print("\nConfusion Matrix:")
print(cm_df)
print(f"\n  Spam correctly caught:     {cm[0][0]}")
print(f"  Spam that slipped through: {cm[0][1]}  (false negatives)")
print(f"  OK flagged as spam:        {cm[1][0]}  (false positives)")
print(f"  OK correctly passed:       {cm[1][1]}")
print("\nSaved model files: vectorizer.pkl, spam_model.pkl")


