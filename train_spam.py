

import argparse
import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: nltk for better preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------
# Text preprocessing utils
# -------------------------
LEMMA = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=True, do_lemmatize=True):
    """
    Basic text cleaning:
      - lowercasing
      - remove email addresses, urls, HTML tags, non-alphanumeric characters
      - optional stopword removal and lemmatization
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove urls
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)
    # keep only letters and numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_lemmatize:
        tokens = [LEMMA.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# -------------------------
# Main training pipeline
# -------------------------
def load_sms_dataset(path):
    """
    Load SMS Spam Collection format (two columns: label \t message)
    If using another CSV, adapt accordingly.
    """
    # Some versions have headerless tab separated file.
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # 0 = ham, 1 = spam
    return df

def train_and_evaluate(data_path, model_out, use_gridsearch=False, random_state=42):
    # Load
    print("Loading dataset from:", data_path)
    df = load_sms_dataset(data_path)
    print("Total samples:", len(df))
    print(df['label'].value_counts())

    # Clean messages
    print("Cleaning text (this may take a while)...")
    df['clean_text'] = df['message'].apply(clean_text)

    X = df['clean_text'].values
    y = df['label'].values

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Pipeline: Tfidf -> Classifier
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2)
    clf = MultinomialNB(alpha=0.5)

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])

    if use_gridsearch:
        print("Running GridSearchCV (this will take longer)...")
        param_grid = {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__alpha': [0.1, 0.5, 1.0]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        print("Training pipeline (TF-IDF + MultinomialNB)...")
        best_model = pipeline
        best_model.fit(X_train, y_train)

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    print("\nEvaluation on test set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'], yticklabels=['ham','spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.splitext(model_out)[0] + "_confusion.png"
    plt.savefig(cm_path)
    print("Saved confusion matrix to:", cm_path)
    plt.close()

    # Save the model (including vectorizer inside pipeline)
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(best_model, model_out)
    print("Saved trained model to:", model_out)

    return best_model

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam classifier")
    parser.add_argument('--data_path', type=str, default='data/SMSSpamCollection',
                        help='Path to SMS Spam data (tab-separated, no header)')
    parser.add_argument('--model_out', type=str, default='models/spam_clf.joblib',
                        help='Path to save trained model')
    parser.add_argument('--grid', action='store_true', help='Run GridSearchCV')
    args = parser.parse_args()

    train_and_evaluate(args.data_path, args.model_out, use_gridsearch=args.grid)
