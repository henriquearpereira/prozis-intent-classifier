import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import unicodedata
import re
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# -- Preprocessing --
def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.category(c) == "Mn")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b\d+\b', '', text)
    stopwords = {
        'a', 'ao', 'aos', 'as', 'da', 'das', 'de', 'do', 'dos', 'e', 'em',
        'na', 'nas', 'no', 'nos', 'o', 'os', 'para', 'por', 'que', 'se',
        'um', 'uma', 'uns', 'umas', 'com', 'como', 'mais', 'mas', 'ou',
        'ser', 'ter', 'esta', 'este', 'isso', 'sua', 'seu', 'dela', 'dele'
    }
    text = " ".join(w for w in text.split() if w not in stopwords and len(w) > 1)
    return text

# -- Load and preprocess data --
def load_and_prepare_dataset(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    df['processed_text'] = df['text'].apply(preprocess_text)
    df = df[df['processed_text'].str.len() > 0]
    return df['processed_text'].values, df['label'].values

# -- Benchmark function with fixed parameter grids --
def benchmark_models(data_path):
    # Load and preprocess
    X, y = load_and_prepare_dataset(data_path)
    
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Fixed GridSearch for SGDClassifier ---
    print("Running GridSearchCV for SGDClassifier...")
    sgd_model = SGDClassifier(max_iter=1000, class_weight='balanced', random_state=42)
    sgd_param_grid = {
        "alpha": [1e-4, 1e-3, 1e-2],
        "penalty": ['l2', 'l1', 'elasticnet'],
        "loss": ['log_loss', 'hinge'],
        "learning_rate": ['optimal'],  # Removed 'invscaling'
    }
    sgd_gs = GridSearchCV(
        sgd_model, 
        sgd_param_grid, 
        cv=skf, 
        scoring='accuracy', 
        n_jobs=-1,
        error_score=np.nan  # Handle errors gracefully
    )
    sgd_gs.fit(X_vec, y_enc)
    print("Best SGDClassifier params:", sgd_gs.best_params_)
    print()

    # --- Fixed GridSearch for LinearSVC ---
    print("Running GridSearchCV for LinearSVC...")
    svc_model = LinearSVC(max_iter=10000, random_state=42)
    svc_param_grid = [
        {
            "C": [0.1, 1, 10],
            "loss": ["hinge"],
            "dual": [True],  # Hinge only works with dual=True
            "penalty": ["l2"]
        },
        {
            "C": [0.1, 1, 10],
            "loss": ["squared_hinge"],
            "dual": [True, False],
            "penalty": ["l2"]
        }
    ]
    svc_gs = GridSearchCV(
        svc_model, 
        svc_param_grid, 
        cv=skf, 
        scoring='accuracy', 
        n_jobs=-1,
        error_score=np.nan  # Handle errors gracefully
    )
    svc_gs.fit(X_vec, y_enc)
    print("Best LinearSVC params:", svc_gs.best_params_)
    print()

    # --- Benchmark models with tuned params ---
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "ComplementNB": ComplementNB(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),  # Removed deprecated parameter
        "SGDClassifier (tuned)": SGDClassifier(max_iter=1000, class_weight='balanced', random_state=42, **sgd_gs.best_params_),
        "LinearSVC (tuned)": LinearSVC(max_iter=10000, random_state=42, **svc_gs.best_params_),
    }

    print("Model                          | Mean Accuracy")
    print("------------------------------------------------")
    for name, model in models.items():
        fold_scores = []
        for train_idx, test_idx in skf.split(X_vec, y_enc):
            X_train, X_test = X_vec[train_idx], X_vec[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_scores.append(accuracy_score(y_test, preds))
        print(f"{name:<30} | {np.mean(fold_scores):.4f}")

# -- Run benchmark --
if __name__ == "__main__":
    data_path = "data/enhanced_intent_dataset.json"
    benchmark_models(data_path)