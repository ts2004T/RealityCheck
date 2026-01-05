# RealityCheck - Fake News Detection

## Overview
RealityCheck is an AI-powered web application that detects whether a news article is REAL or FAKE using Machine Learning. It analyzes the text content of the article and classifies it based on linguistic patterns learned from a dataset of labeled news.

## Problem Statement
The spread of fake news is a growing concern in the digital age. Misinformation can manipulate public opinion, influence elections, and cause social unrest. This project aims to provide a simple, accessible tool to help users verify the authenticity of news articles.

## Tech Stack
- **Python 3**: Core programming language.
- **Flask**: Web framework for the backend.
- **Scikit-learn**: Machine Learning library (Logistic Regression, TF-IDF).
- **Pandas & NumPy**: Data manipulation and analysis.
- **NLTK**: Natural Language Processing (Tokenization, Lemmatization, Stopwords).
- **Joblib**: Model persistence (saving/loading trained models).
- **HTML/CSS**: Frontend user interface.

## How it Works (ML Pipeline)
1.  **Data Ingestion**: Loads labeled news data from `dataset/news.csv`.
2.  **Preprocessing**:
    - Converts text to lowercase.
    - Removes punctuation and numbers.
    - Removes stopwords (common words like "the", "is").
    - Lemmatizes words (converts "running" to "run").
3.  **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors.
4.  **Model Training**: Trains a **Logistic Regression** model on the vectorized data.
5.  **Inference**: The Flask app loads the saved model and vectorizer to predict new inputs.

## How to Run on Replit
The project is configured to run automatically.

1.  **Train the Model**: The model is pre-trained, but if you change the dataset, run:
    ```bash
    python train_model.py
    ```
2.  **Start the App**: Click the **Run** button.
    - This executes `python app.py` (via the configured entry point).
3.  **Use the App**:
    - Open the web view.
    - Paste a news article into the text area.
    - Click "Check Authenticity".
    - View the result ("REAL NEWS ✅" or "FAKE NEWS ❌").

## Future Improvements
- Integrate a larger, real-world dataset (e.g., Kaggle Fake News Dataset).
- Implement deep learning models (LSTM, BERT) for better accuracy.
- Add URL scraping to check news directly from links.
- Add a "confidence score" to the prediction.

## Project Structure
```
RealityCheck/
├── app.py                     # Flask application
├── model/                     # Saved ML assets
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── dataset/
│   └── news.csv               # Dataset
├── templates/
│   └── index.html             # UI
├── static/
│   └── style.css              # Styling
├── train_model.py             # Training script
└── README.md                  # Documentation
```
