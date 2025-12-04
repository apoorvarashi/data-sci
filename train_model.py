import pandas as pd
import numpy as np
import re
import string
import nltk
import sys
import joblib
import warnings
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# --- NLTK Setup ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.1')
try:
    SentimentIntensityAnalyzer().polarity_scores("test")
except LookupError:
    nltk.download('vader_lexicon')


# --- Configuration ---
DATA_FILE = 'sample.csv'
COLUMNS = ['ID', 'Entity', 'Sentiment', 'Tweet']

# --- 1. Data Loading ---
print("1. Loading Data...")
try:
    print(f"Attempting to load data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS, encoding='latin-1')
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{DATA_FILE}' was not found. Please ensure it is in the same directory.")
    sys.exit(1)

# Basic cleaning
df = df[['Sentiment', 'Tweet']].copy()
df.dropna(subset=['Tweet'], inplace=True)
df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])].copy()
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded. Total clean samples: {len(df)}")
print("-" * 50)


# --- 2. Text Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and prepares text for vectorization."""
    if isinstance(text, float) or pd.isna(text): 
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'@\w+|#', '', text) 
    # We are removing punctuation only to clean up word tokens for TFIDF, 
    # but VADER (used below) is specifically designed to handle and benefit from it.
    text = re.sub(r'\d+', '', text) 
    
    # Do NOT remove all punctuation here, as VADER relies on it for intensity!
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] 
    
    return " ".join(tokens)

print("2. Preprocessing Text and Extracting VADER Features...")
df['Clean_Tweet'] = df['Tweet'].apply(preprocess_text)

# --- VADER Feature Extraction ---
analyzer = SentimentIntensityAnalyzer()
vader_scores = df['Tweet'].apply(lambda x: analyzer.polarity_scores(str(x)))
vader_df = pd.json_normalize(vader_scores)
df = pd.concat([df, vader_df], axis=1)
print("VADER features extracted: neg, neu, pos, compound.")
print("-" * 50)


# --- 3. Feature Extraction (TF-IDF with VADER features) ---
X = df['Clean_Tweet']
y = df['Sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Feature Extraction (TF-IDF + VADER features)...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)) 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Extract VADER features for train/test sets
vader_cols = ['neg', 'neu', 'pos', 'compound']
X_train_vader = df.loc[X_train.index, vader_cols].values
X_test_vader = df.loc[X_test.index, vader_cols].values

# Combine TFIDF matrix (sparse) with VADER features (dense)
X_train_vec = hstack([X_train_tfidf, X_train_vader])
X_test_vec = hstack([X_test_tfidf, X_test_vader])

print(f"Combined feature dimension: {X_train_vec.shape[1]}")
print("-" * 50)


# --- 4. Model Training and Evaluation ---
results = {}
best_accuracy = 0.0

# Remaining models for training: MNB (TFIDF Only), Logistic Regression, Decision Tree, Random Forest
print("4. Training and Evaluating Models (MNB, LR, Decision Tree, Random Forest)...")

# --- Model 1: Multinomial Naive Bayes (Baseline) ---
print("\nTraining Multinomial Naive Bayes (Baseline)...")
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train) # MNB cannot handle negative numbers, so we train it ONLY on TFIDF
y_pred_mnb = mnb.predict(X_test_tfidf)
acc_mnb = accuracy_score(y_test, y_pred_mnb)
results['Multinomial Naive Bayes (TFIDF Only)'] = {'accuracy': acc_mnb, 'report': classification_report(y_test, y_pred_mnb, output_dict=True)}
print(f"MNB Accuracy: {acc_mnb:.4f}")

# --- Model 2: Logistic Regression (Hyperparameter Tuned) ---
print("\nTraining Tuned Logistic Regression (Combined Features)...")
lr = LogisticRegression(max_iter=5000, random_state=42)
param_grid = {'C': [1, 10]} 
grid_search_lr = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
grid_search_lr.fit(X_train_vec, y_train)

best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_vec)
acc_lr = accuracy_score(y_test, y_pred_lr)
results['Tuned Logistic Regression'] = {'accuracy': acc_lr, 'report': classification_report(y_test, y_pred_lr, output_dict=True)}
print(f"Tuned LR Accuracy: {acc_lr:.4f}")

# --- Model 3: Decision Tree Classifier (Combined Features) ---
print("\nTraining Decision Tree Classifier (Combined Features)...")
# Decision Tree is a non-linear model, suitable for mixed feature types.
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_vec, y_train)
y_pred_dtc = dtc.predict(X_test_vec)
acc_dtc = accuracy_score(y_test, y_pred_dtc)
results['Decision Tree Classifier'] = {'accuracy': acc_dtc, 'report': classification_report(y_test, y_pred_dtc, output_dict=True)}
print(f"Decision Tree Accuracy: {acc_dtc:.4f}")


# --- Model 4: Random Forest Classifier (Combined Features) ---
print("\nTraining Random Forest Classifier (Combined Features)...")
# Random Forest is an ensemble of Decision Trees that reduces overfitting.
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rfc.fit(X_train_vec, y_train)
y_pred_rfc = rfc.predict(X_test_vec)
acc_rfc = accuracy_score(y_test, y_pred_rfc)
results['Random Forest Classifier'] = {'accuracy': acc_rfc, 'report': classification_report(y_test, y_pred_rfc, output_dict=True)}
print(f"Random Forest Accuracy: {acc_rfc:.4f}")


# --- 5. Summary and Saving ---
print("-" * 50)
print("5. Summary and Persistence...")

# Determine the overall best model and assign the correct object
# List of all combined-feature models for deployment (excluding SVC and XGBoost)
combined_models = {
    'Tuned Logistic Regression': best_lr,
    'Decision Tree Classifier': dtc,
    'Random Forest Classifier': rfc,
}
combined_results = {k: v for k, v in results.items() if k in combined_models}

if not combined_results:
    # Fallback if no combined models trained successfully
    best_deployment_model_name = 'Multinomial Naive Bayes (TFIDF Only)'
    best_model = mnb
    best_accuracy = results[best_deployment_model_name]['accuracy']
else:
    best_deployment_model_name = max(combined_results, key=lambda name: combined_results[name]['accuracy'])
    best_model = combined_models[best_deployment_model_name]
    best_accuracy = combined_results[best_deployment_model_name]['accuracy']

# Save all assets required by the Streamlit app
joblib.dump(results, 'model_comparison_results.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Save the Best Model
joblib.dump(best_model, 'best_model.pkl')

print(f"\n--- Model Comparison Results (All Models) ---")
for name, res in results.items():
    print(f"-> {name}: Accuracy = {res['accuracy']:.4f}")

print(f"\n✅ Training complete.")
print(f"✅ Best Model for deployment: {best_deployment_model_name} with Accuracy: {best_accuracy:.4f}")
print("✅ Assets saved: 'tfidf_vectorizer.pkl', 'best_model.pkl', 'model_comparison_results.pkl'")
print("\nNOW, RUN: streamlit run app.py")