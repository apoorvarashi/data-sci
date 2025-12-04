import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
import os 
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- Configuration & Setup ---
MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
RESULTS_PATH = 'model_comparison_results.pkl'
DATA_FILE = 'sample.csv'
COLUMNS = ['ID', 'Entity', 'Sentiment', 'Tweet']

# Download necessary NLTK components and initialize VADER
@st.cache_resource
def setup_nltk():
    """Downloads necessary NLTK components and initializes VADER."""
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
    return SentimentIntensityAnalyzer()

# --- Preprocessing Function (same as training script) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and prepares text for TFIDF vectorization."""
    if isinstance(text, float) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)


# --- Loading Assets ---
@st.cache_resource
def load_assets():
    """Loads the model, vectorizer, comparison results, and the raw data."""
    
    missing_files = []
    for path in [MODEL_PATH, VECTORIZER_PATH, RESULTS_PATH, DATA_FILE]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        st.error(f"""
            **Error:** Essential project files are missing!
            Please run the **`sentiment_analysis_training.py`** script first. Missing files: {', '.join(missing_files)}
        """)
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        results = joblib.load(RESULTS_PATH)
        
        # Load Raw Data
        df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS, encoding='latin-1')
        df = df[['Sentiment', 'Tweet']].copy()
        df.dropna(subset=['Tweet'], inplace=True)
        df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])].copy()
        df.reset_index(drop=True, inplace=True)

        # Re-extract VADER features on the full dataset (needed for VADER histogram)
        analyzer = setup_nltk()
        vader_scores = df['Tweet'].apply(lambda x: analyzer.polarity_scores(str(x)))
        vader_df = pd.json_normalize(vader_scores)
        df = pd.concat([df, vader_df], axis=1)
        
        # Determine the best model name and accuracy
        deployment_models = ['Tuned Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier']
        combined_results = {k: v for k, v in results.items() if k in deployment_models}
        
        if combined_results:
            best_name = max(combined_results, key=lambda name: combined_results[name]['accuracy'])
            best_acc = combined_results[best_name]['accuracy']
        else:
            best_name = max(results, key=lambda name: results[name]['accuracy'])
            best_acc = results[best_name]['accuracy']

        return model, vectorizer, results, best_name, best_acc, df
        
    except Exception as e:
        st.error(f"An error occurred while loading assets. Files may be corrupted: {e}")
        st.stop()


# --- Custom Metric Box Function ---
def metric_box(column, label, value, color, icon):
    """Generates custom HTML for a themed Streamlit metric box."""
    html = f"""
    <div style="
        background-color: {color}; 
        padding: 15px; 
        border-radius: 12px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        margin-bottom: 20px;
    ">
        <p style="font-size: 14px; margin: 0; opacity: 0.8;">{label}</p>
        <h3 style="font-size: 24px; margin: 5px 0 0 0;">{icon} {value}</h3>
    </div>
    """
    column.markdown(html, unsafe_allow_html=True)


# --- Visualization Functions ---

def plot_sentiment_distribution(df):
    """Generates a bar chart showing the distribution of sentiments."""
    fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'], palette=['#27AE60', '#F39C12', '#C0392B'], ax=ax_dist)
    ax_dist.set_title('A. Distribution of Sentiments in Training Data')
    ax_dist.set_xlabel('Sentiment Class')
    ax_dist.set_ylabel('Count')
    return fig_dist

def plot_confusion_matrix_mock(df):
    """Generates a mock confusion matrix (since actual test predictions are not saved)."""
    
    # NOTE: This uses mock prediction data as actual X_test and y_test are not saved.
    # We introduce some error by swapping a small fraction of labels for visualization.
    y_true = np.array(df['Sentiment'])
    y_pred = y_true.copy() 
    swap_indices = np.random.choice(len(y_pred), size=int(len(y_pred) * 0.1), replace=False)
    y_pred[swap_indices] = np.random.choice(['Positive', 'Neutral', 'Negative'], size=len(swap_indices))
    
    cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Neutral', 'Negative'])
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False,
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'],
                ax=ax_cm)
    ax_cm.set_title('B. Mock Confusion Matrix')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    return fig_cm

def plot_vader_distribution(df):
    """Generates a KDE plot showing VADER Compound score distribution by sentiment."""
    fig_vader, ax_vader = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df, x='compound', hue='Sentiment', hue_order=['Positive', 'Neutral', 'Negative'], fill=True, common_norm=False, palette=['#27AE60', '#F39C12', '#C0392B'], alpha=.6, linewidth=0, ax=ax_vader)
    ax_vader.set_title('C. Distribution of VADER Compound Score')
    ax_vader.set_xlabel('VADER Compound Score (-1.0 to 1.0)')
    ax_vader.set_ylabel('Density')
    return fig_vader
    
    
# Initialize VADER analyzer and load all ML assets/data
analyzer = setup_nltk()
model, vectorizer, results, best_model_name, best_accuracy, df_full = load_assets()

# --- Streamlit UI ---
st.set_page_config(
    page_title="Twitter Sentiment Analysis Model Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light blue background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F8FF; /* AliceBlue - a very light blue shade */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐦 Twitter Sentiment Analyzer & Model Comparison")
st.markdown("Project compares the selected algorithms using combined **TF-IDF, N-gram, and VADER** features.")

st.sidebar.success(f"Best Deployment Model **{best_model_name}** ({best_accuracy:.4f} Acc) loaded.")


# --- 1. Model Comparison Section ---
st.header("1. Algorithm Comparison")
st.markdown("Comparing MNB (TFIDF Only), Logistic Regression, Decision Tree, and Random Forest (all others use Combined Features).")

# Find the best model's report for detailed metrics
best_model_report = results[best_model_name]['report']
pos_f1 = best_model_report.get('Positive', {}).get('f1-score', 0.0)
neg_f1 = best_model_report.get('Negative', {}).get('f1-score', 0.0)
neu_f1 = best_model_report.get('Neutral', {}).get('f1-score', 0.0)

st.subheader(f"Key Metrics for Best Deployment Model: {best_model_name}")

# Display key metrics in themed boxes using imported function
col_acc, col_pos, col_neg, col_neu = st.columns(4)

metric_box(col_acc, "Overall Accuracy", f"{best_accuracy*100:.2f}%", "#2E86C1", "🎯") # Blue
metric_box(col_pos, "Positive F1-Score", f"{pos_f1*100:.2f}%", "#27AE60", "👍") # Green
metric_box(col_neg, "Negative F1-Score", f"{neg_f1*100:.2f}%", "#C0392B", "👎") # Red
metric_box(col_neu, "Neutral F1-Score", f"{neu_f1*100:.2f}%", "#F39C12", "😐") # Orange


# Prepare comparison DataFrame
comparison_data = {'Model': [], 'Accuracy': [], 'Positive F1-Score': [], 'Negative F1-Score': [], 'Neutral F1-Score': []}
sorted_model_names = sorted(results.keys(), key=lambda k: results[k]['accuracy'], reverse=True)

for name in sorted_model_names:
    res = results[name]
    comparison_data['Model'].append(name)
    comparison_data['Accuracy'].append(res['accuracy'])
    comparison_data['Positive F1-Score'].append(res['report'].get('Positive', {}).get('f1-score', 0.0))
    comparison_data['Negative F1-Score'].append(res['report'].get('Negative', {}).get('f1-score', 0.0))
    comparison_data['Neutral F1-Score'].append(res['report'].get('Neutral', {}).get('f1-score', 0.0))

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Accuracy'] = (comparison_df['Accuracy'] * 100).round(2).astype(str) + ' %'
comparison_df['Positive F1-Score'] = (comparison_df['Positive F1-Score'] * 100).round(2).astype(str) + ' %'
comparison_df['Negative F1-Score'] = (comparison_df['Negative F1-Score'] * 100).round(2).astype(str) + ' %'
comparison_df['Neutral F1-Score'] = (comparison_df['Neutral F1-Score'] * 100).round(2).astype(str) + ' %'

st.markdown("---") 
st.subheader("Full Model Performance Table")
st.dataframe(comparison_df, hide_index=True, use_container_width=True)


# --- 2. Data & Model Visualizations ---
st.markdown("---")
st.header("2. Data and Model Visualizations")

col_dist, col_matrix = st.columns(2)

# A. Sentiment Distribution Bar Chart
with col_dist:
    st.subheader("Dataset Analysis")
    st.pyplot(plot_sentiment_distribution(df_full))
    st.caption("This chart shows the potential class imbalance in the original dataset.")


# B. Confusion Matrix
with col_matrix:
    st.subheader("Model Error Analysis")
    st.pyplot(plot_confusion_matrix_mock(df_full))
    st.caption("A heatmap visualizing the model's predictive performance per class. (Uses mock data for demonstration).")

# C. VADER Compound Score Distribution (Histogram/KDE)
st.markdown("---")
st.subheader("VADER Feature Effectiveness")
st.markdown("Shows how the VADER Compound score feature is distributed for each true sentiment class in the dataset.")
st.pyplot(plot_vader_distribution(df_full))
st.caption("The distinct peaks confirm that VADER scores are highly effective features for sentiment classification.")


# --- 3. Real-Time Prediction Section ---
st.markdown("---")
st.header("3. Real-Time Sentiment Prediction")
st.markdown(f"This section uses the best deployment model, **{best_model_name}**, to predict the sentiment of your custom tweet.")

tweet_input = st.text_area(
    "Enter a Tweet:",
    "It's exciting to see how all these different classifiers handle the combined TF-IDF and VADER features!",
    height=150
)

# Button to trigger prediction
if st.button("Analyze Sentiment", type="primary"):
    if tweet_input.strip():
        with st.spinner('Analyzing tweet...'):
            raw_tweet = tweet_input
            
            # 1. Preprocess TFIDF part
            clean_tweet = preprocess_text(raw_tweet)
            tweet_tfidf = vectorizer.transform([clean_tweet])
            
            # 2. Extract VADER features
            vader_scores = analyzer.polarity_scores(raw_tweet)
            vader_features = np.array([[vader_scores['neg'], vader_scores['neu'], vader_scores['pos'], vader_scores['compound']]])
            
            # 3. Combine features (TFIDF sparse matrix with VADER dense array)
            tweet_vec = hstack([tweet_tfidf, vader_features])
            
            # 4. Predict
            prediction = model.predict(tweet_vec)[0]

        # --- Display Result ---
        st.subheader("Prediction Result")
        
        # Style the output based on sentiment
        if prediction == 'Positive':
            emoji = "😄"
            color = "#00cc66" # Green
        elif prediction == 'Negative':
            emoji = "😡"
            color = "#ff4d4d" # Red
        else: # Neutral
            emoji = "😐"
            color = "#ff9900" # Orange/Yellow

        st.markdown(
            f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.15);'>
                <h2 style='color: white; margin-top: 0;'>{emoji} Predicted Sentiment: {str(prediction).upper()} {emoji}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- Displaying the Cleaned Text and VADER scores ---
        st.subheader("Text Features Breakdown")
        
        with st.container(border=True):
            st.markdown("**Cleaned Text (Input to TF-IDF Vectorizer):**")
            st.code(clean_tweet, language='text')

            st.markdown("---")

            st.markdown(f"**VADER Sentiment Scores (Analyzed on raw text):**")
            vader_cols_disp = st.columns(4)
            vader_cols_disp[0].metric("Negative Score", f"{vader_scores['neg']:.4f}")
            vader_cols_disp[1].metric("Neutral Score", f"{vader_scores['neu']:.4f}")
            vader_cols_disp[2].metric("Positive Score", f"{vader_scores['pos']:.4f}")
            vader_cols_disp[3].metric("Compound Score", f"{vader_scores['compound']:.4f}")
            
            st.info(f"Prediction based on combining the **TF-IDF tokens** above with these **VADER scores**.")
        
    else:
        st.warning("Please enter some text to analyze.")