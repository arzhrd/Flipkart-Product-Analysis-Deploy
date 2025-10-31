# ==========================================
# FLIPKART REVIEW ANALYZER - STREAMLIT APP
# ==========================================
import streamlit as st
import joblib
import pandas as pd
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Set up NLTK components
nltk.download('stopwords')
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words('english'))

@st.cache_resource
def load_artifacts():
    """
    Load the saved model, vectorizer, and label encoder.
    The @st.cache_resource decorator ensures this runs only once.
    """
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, vectorizer, le
    except FileNotFoundError:
        st.error("Model files not found! Please make sure 'sentiment_model.pkl', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are in the same folder as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load the models
model, vectorizer, le = load_artifacts()

# ==========================================
# 2. PREPROCESSING FUNCTION
# ==========================================

@st.cache_data
def clean(text):
    """
    The exact same cleaning pipeline from your notebook.
    Using @st.cache_data to speed up processing of repeated reviews.
    """
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# ==========================================
# 3. WEB SCRAPING FUNCTION (PLACEHOLDER)
# ==========================================

def scrape_flipkart_reviews(url):
    """
    *** THIS IS A PLACEHOLDER SIMULATION ***
    Real web scraping is complex. This function simulates
    fetching reviews to let us test the ML pipeline.
    """
    with st.spinner(f"Scraping reviews from {url}... (This is a 3-second simulation)"):
        time.sleep(3)
    
    # Return a list of dummy reviews to test the model
    return [
        "The product is absolutely amazing! Best purchase of the year.",
        "I love it. The quality is top-notch and it was delivered fast.",
        "Worst product I have ever bought. It broke after just one day.",
        "This is a complete waste of money. Do not recommend.",
        "It's an okay product. Not great, not terrible. Just average.",
        "Good value for the price. Satisfied with my purchase.",
        "Terrible customer service and the product was defective.",
        "Five stars! Will definitely buy from this seller again.",
        "The item I received was the wrong color. Very disappointed.",
        "It's decent. Gets the job done."
    ]

# ==========================================
# 4. STREAMLIT UI & MAIN LOGIC
# ==========================================

st.set_page_config(page_title="Flipkart Review Analyzer", page_icon="🛍️", layout="wide")
st.title("🛍️ Flipkart Product Review Analyzer")
st.markdown("Enter a Flipkart product URL to analyze customer sentiment and see if most consumers are satisfied.")

# Only proceed if models were loaded successfully
if model and vectorizer and le:
    url = st.text_input("Enter a Flipkart Product URL:", placeholder="https://www.flipkart.com/...")

    if st.button("Analyze Sentiment", type="primary"):
        if not url.startswith("https://www.flipkart.com/"):
            st.error("Please enter a valid Flipkart URL.")
        else:
            # --- 1. Scrape ---
            reviews_list = scrape_flipkart_reviews(url)
            
            if not reviews_list:
                st.warning("Could not find any reviews for this product.")
            else:
                # --- 2. Process & Predict ---
                st.subheader(f"Analyzing {len(reviews_list)} reviews...")
                predictions = []
                with st.spinner("Running sentiment analysis model..."):
                    for review_text in reviews_list:
                        cleaned_review = clean(review_text)
                        
                        # Only predict if cleaning doesn't result in an empty string
                        if cleaned_review:
                            vectorized_review = vectorizer.transform([cleaned_review])
                            prediction_int = model.predict(vectorized_review)[0]
                            sentiment = le.inverse_transform([prediction_int])[0]
                            predictions.append(sentiment)
                
                if not predictions:
                    st.warning("After cleaning, no valid review text was found to analyze.")
                else:
                    # --- 3. Aggregate & Display Results ---
                    sentiment_counts = pd.Series(predictions).value_counts()
                    
                    # Ensure all categories exist for the chart
                    if "Positive" not in sentiment_counts: sentiment_counts["Positive"] = 0
                    if "Negative" not in sentiment_counts: sentiment_counts["Negative"] = 0
                    if "Neutral" not in sentiment_counts: sentiment_counts["Neutral"] = 0
                    
                    # --- FINAL VERDICT ---
                    st.subheader("Final Verdict")
                    positive_count = sentiment_counts.get("Positive", 0)
                    negative_count = sentiment_counts.get("Negative", 0)
                    
                    if positive_count > negative_count:
                        st.success(f"**Overall: POSITIVE** ({positive_count} vs {negative_count} negative)")
                        st.markdown("### ✅ **Most consumers are satisfied with this product.**")
                    elif negative_count > positive_count:
                        st.error(f"**Overall: NEGATIVE** ({negative_count} vs {positive_count} positive)")
                        st.markdown("### ❌ **Most consumers are *not* satisfied with this product.**")
                    else:
                        st.warning(f"**Overall: MIXED / NEUTRAL** ({positive_count} positive, {negative_count} negative)")
                        st.markdown("### 🤷 **Consumer sentiment is mixed.**")

                    # --- Charts ---
                    st.subheader("Sentiment Breakdown")
                    st.bar_chart(sentiment_counts, color=["#2ecc71", "#e74c3c", "#95a5a6"]) # Green, Red, Gray

                    # --- Show a sample of reviews ---
                    st.subheader("Sampled Reviews (from simulation)")
                    sample_df = pd.DataFrame({
                        'Review Text': reviews_list,
                        'Predicted Sentiment': predictions
                    })
                    st.dataframe(sample_df)

    st.info("**Disclaimer:** The review 'scraping' is currently a simulation. This demo uses 10 pre-programmed reviews to show the complete ML pipeline in action.")
else:
    st.stop()
