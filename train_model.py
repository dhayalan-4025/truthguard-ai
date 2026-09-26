"""
train_model.py
---------------
Builds a small demo dataset (WhatsApp forwards, fake news headlines, phishing/spam
emails, and normal everyday messages) and trains a TF-IDF + Multinomial Naive Bayes
text classifier for TruthGuard AI.

IMPORTANT: The dataset here is small and hand-written for demo purposes only.
For real accuracy, replace `load_dataset()` with a proper labelled dataset
(e.g. a CSV of thousands of real spam/fake/normal messages) — see README.md.

Run:
    python train_model.py
Produces:
    model/vectorizer.pkl
    model/classifier.pkl
"""

import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation noise, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    """
    Returns (texts, labels) where label is one of:
        "suspicious"  -> fake news / spam / phishing / scam forward
        "normal"      -> genuine, everyday message
    """
    suspicious = [
        "CONGRATULATIONS! You have WON a lottery of Rs 25,00,000. Click here to claim your prize now!",
        "Forward this message to 10 people or you will have bad luck for 7 years",
        "URGENT: Your bank account will be BLOCKED today. Verify your KYC by clicking this link immediately",
        "Breaking: Scientists confirm that drinking hot water cures all types of cancer, share before it gets deleted",
        "Get a free iPhone 15 just by filling this survey, limited offer only for today",
        "Your Amazon account has been suspended. Login here to reactivate within 24 hours",
        "PM announces free gold coins for every citizen, click link to register your Aadhaar",
        "This new government scheme gives 5 lakh rupees to every unemployed youth, apply now before deadline",
        "WARNING: New WhatsApp update will charge you Rs 500 unless you forward this message immediately",
        "You have been selected for a cash reward of $10000, send your bank details to claim",
        "Miracle fruit found in Amazon forest cures diabetes permanently overnight, doctors shocked",
        "Alert: Eating this vegetable everyday can shrink kidney stones in 3 days, must share",
        "Your parcel could not be delivered, pay Rs 49 customs fee at this link to reschedule",
        "Job offer: Earn Rs 5000 daily by just liking Youtube videos from home, no investment",
        "Aliens spotted near Mumbai airport last night, government hiding the truth say experts",
        "This bank is going to shut down forever from tomorrow, withdraw all your cash today",
        "Click this link to update your SIM card KYC or it will be deactivated within 2 hours",
        "Free recharge of Rs 199 for all Jio users, offer valid only for next 30 minutes, click now",
        "Share this post and Elon Musk will personally send you a free Tesla car",
        "Old Rs 10 coin can be sold for Rs 5 lakh, sell yours through this website today",
        "Whatsapp is going to become paid from next month unless you forward this to 5 groups",
        "Your electricity connection will be disconnected tonight, pay pending bill via this link",
        "New RBI rule: all ATM cards will stop working from Monday, visit this site to update",
        "You've been chosen to test our new smartwatch for free, just pay Rs 99 shipping",
        "Doctors don't want you to know this one trick to lose 10kg in a week, click to read",
        "Modi government to give free laptops to all students, register with your Aadhaar number here",
        "This hospital is selling human organs illegally, watch shocking video before it is banned",
        "Your Netflix subscription payment failed, update card details immediately at this link",
        "Earn Bitcoin instantly with this new trick, invest Rs 1000 and get Rs 10000 in a week",
        "Fake currency notes are being circulated in your city, forward to warn everyone urgently",
    ]

    normal = [
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "The quarterly sales report has been shared in the team drive, please review before Monday",
        "Happy birthday! Hope you have a wonderful day with family and friends",
        "Can you send me the notes from today's class, I missed the first half",
        "The weather looks good this weekend, shall we plan the trek?",
        "Reminder: the electricity bill payment is due on the 15th of this month",
        "Great job on the presentation today, the client seemed really impressed",
        "Let's catch up over coffee sometime next week, it's been a while",
        "The train to Chennai is delayed by 20 minutes according to the app",
        "Please find attached the invoice for last month's services",
        "Congratulations on your promotion, well deserved after all the hard work",
        "I have submitted the assignment, let me know if you need any changes",
        "The doctor's appointment is confirmed for Thursday at 4pm",
        "Thanks for helping me move the furniture yesterday, really appreciate it",
        "Our flight departs at 6am so let's leave for the airport by 4",
        "The new library book policy allows up to 5 books for two weeks",
        "Mom said dinner will be ready by 8, don't be late",
        "The project deadline has been extended to next Friday, please plan accordingly",
        "I watched a really good documentary on ocean life last night",
        "Can we reschedule our meeting to 3pm instead of 2pm today?",
        "The cricket match starts at 7pm, want to watch it together?",
        "Please remember to bring your ID card for tomorrow's exam",
        "The plumber will come to fix the leak on Wednesday morning",
        "I really enjoyed the book you recommended, thank you for the suggestion",
        "Team lunch is planned for Friday afternoon at the new restaurant downtown",
        "The college fest committee meeting is scheduled for 5pm in room 204",
        "Traffic on the highway is heavy today because of ongoing roadwork",
        "Please review the attached document and share your feedback by tomorrow",
        "It was nice catching up with you at the reunion last weekend",
        "The gym is closed for maintenance this Sunday, opens again on Monday",
    ]

    texts = suspicious + normal
    labels = ["suspicious"] * len(suspicious) + ["normal"] * len(normal)
    return texts, labels


def main():
    texts, labels = load_dataset()
    cleaned = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("Evaluation on held-out demo data:")
    print(classification_report(y_test, preds))

    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print(f"\nSaved vectorizer.pkl and classifier.pkl to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
