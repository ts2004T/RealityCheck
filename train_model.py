import pandas as pd
import numpy as np
import pickle
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Tokenization (split by space)
    tokens = text.split()
    # 4. Remove stopwords and 5. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def train():
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset/news.csv')
    except FileNotFoundError:
        print("Error: dataset/news.csv not found.")
        return

    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Feature Extraction
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save Model and Vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    train()
