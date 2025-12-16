import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add src to path to import DatasetLoader
sys.path.append(os.path.join(os.path.dirname(__file__)))
from etl_pipeline import DatasetLoader

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def clean_sentiment(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https.?:/\/\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def train_model():
    print("Loading dataset...")
    # Go up one level to find the csv
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'flipkart_product.csv')
    loader = DatasetLoader(csv_path)
    df = loader.load()
    
    if df is None:
        print("Failed to load dataset.")
        return

    print("Columns:", df.columns)
    
    # Map Ratings to Sentiment
    # Assuming 'Rate' column exists vs 'rating'
    rate_col = 'Rate' if 'Rate' in df.columns else 'rate'
    
    # Clean rate column (sometimes it has strings?)
    # The head command showed "5", "3" etc.
    df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
    df = df.dropna(subset=[rate_col])
    
    def get_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating <= 2:
            return "Negative"
        else:
            return "Neutral"
            
    df['sentiment'] = df[rate_col].apply(get_sentiment)
    
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Preprocess Text
    text_col = 'Review' if 'Review' in df.columns else 'review'
    # Fallback
    if text_col not in df.columns:
        text_col = df.columns[3] # Index 3 from head output
        
    print(f"Using text column: {text_col}")
    
    print("Cleaning text (this may take a while)...")
    df['clean_text'] = df[text_col].apply(clean_sentiment)
    
    # Split
    X = df['clean_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Train
    print("Training Model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train_enc)
    
    # Evaluate
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test_enc, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))
    
    # Save
    print("Saving artifacts...")
    project_root = os.path.dirname(os.path.dirname(__file__))
    joblib.dump(model, os.path.join(project_root, 'sentiment_model.pkl'))
    joblib.dump(vectorizer, os.path.join(project_root, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(project_root, 'label_encoder.pkl'))
    print("Done!")

if __name__ == "__main__":
    train_model()
