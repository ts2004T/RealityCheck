from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Initialize Flask app
app = Flask(__name__)

# Load Model and Vectorizer
MODEL_PATH = 'model/fake_news_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

model = None
vectorizer = None

# Ensure NLTK resources are available (for runtime)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_resources():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Model or Vectorizer not found. Please run train_model.py first.")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form['text']
        
        if not model or not vectorizer:
            load_resources()
            
        if model and vectorizer and input_text:
            clean_text = preprocess_text(input_text)
            features = vectorizer.transform([clean_text]).toarray()
            result = model.predict(features)[0]
            
            # 1 = REAL, 0 = FAKE
            if result == 1:
                prediction = "REAL NEWS ✅"
            else:
                prediction = "FAKE NEWS ❌"
    
    return render_template('index.html', prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    load_resources()
    app.run(host='0.0.0.0', port=5000, debug=True)
