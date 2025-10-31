# ==========================================
# CUSTOMER SENTIMENT ANALYZER - STREAMLIT APP
# (Flipkart Theme Version)
# ==========================================
import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ==========================================
# 1. SETUP & MODEL LOADING (No Change)
# ==========================================

# Set up NLTK components
@st.cache_data
def load_nltk_data():
    """Downloads NLTK data and returns stopwords and stemmer."""
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    return set(stopwords.words('english')), SnowballStemmer("english")

stopword, stemmer = load_nltk_data()

@st.cache_resource
def load_artifacts():
    """
    Load the saved model, vectorizer, and label encoder.
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
# 2. PREPROCESSING FUNCTION (No Change)
# ==========================================

@st.cache_data
def clean(text):
    """
    The exact same cleaning pipeline from your notebook.
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
# 3. STREAMLIT UI & CUSTOM FLIPKART STYLING
# ==========================================

st.set_page_config(page_title="Flipkart Sentiment Analyzer", page_icon="🛒", layout="wide")

# --- NEW STYLING CODE ---
# We'll use Flipkart's brand colors and font
FLIPKART_BLUE = "#2874f0"
FLIPKART_BACKGROUND = "#f1f3f6"
FLIPKART_TEXT = "#212121"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body, [class*="st-"] {{
    font-family: 'Roboto', sans-serif;
}}

/* Main app background */
.stApp {{
    background-color: {FLIPKART_BACKGROUND};
}}

/* Main title */
h1 {{
    color: {FLIPKART_BLUE};
    font-weight: 700;
}}

/* Subheaders (e.g., "Final Verdict") */
h2 {{
    color: {FLIPKART_TEXT};
}}

/* "Analyze Sentiment" button */
.stButton > button {{
    background-color: {FLIPKART_BLUE};
    color: white;
    border: none;
    border-radius: 2px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
}}
.stButton > button:hover {{
    background-color: #1a5bb9; /* A darker blue for hover */
    color: white;
    border: none;
}}

/* Text area */
.stTextArea textarea {{
    border: 1px solid #c2c2c2;
    background-color: #ffffff;
    font-family: 'Roboto', sans-serif;
    border-radius: 2px;
}}
</style>
""", unsafe_allow_html=True)
# --- END OF STYLING CODE ---


st.title("Flipkart Sentiment Analyzer")
st.markdown("Paste in customer reviews (one review per line) to analyze overall sentiment.")

# ==========================================
# 4. MAIN LOGIC (No Change)
# ==========================================

# Only proceed if models were loaded successfully
if model and vectorizer and le:
    
    # Define a placeholder for the text area
    review_input_placeholder = (
        "This product is amazing!\n"
        "Worst purchase ever. I'm returning it.\n"
        "It's okay, not great but not terrible.\n"
        "I love the color and the quality."
    )
    
    # === INPUT CHANGE: Use st.text_area for multi-line input ===
    user_reviews = st.text_area(
        "Paste reviews here (one review per line):",
        placeholder=review_input_placeholder,
        height=250
    )

    if st.button("Analyze Sentiment", type="primary"):
        
        # === LOGIC CHANGE: Split text area input into a list ===
        if not user_reviews.strip():
            st.error("Please enter at least one review.")
        else:
            # Split by newline and remove any empty strings or whitespace-only lines
            reviews_list = [review.strip() for review in user_reviews.split('\n') if review.strip()]
            
            if not reviews_list:
                st.error("Please enter valid review text.")
            else:
                # --- 2. Process & Predict ---
                st.subheader(f"Analyzing {len(reviews_list)} reviews...")
                predictions = []
                valid_reviews_for_df = [] # To store the original text of valid reviews
                
                with st.spinner("Running sentiment analysis model..."):
                    for review_text in reviews_list:
                        cleaned_review = clean(review_text)
                        
                        # Only predict if cleaning doesn't result in an empty string
                        if cleaned_review:
                            vectorized_review = vectorizer.transform([cleaned_review])
                            prediction_int = model.predict(vectorized_review)[0]
                            sentiment = le.inverse_transform([prediction_int])[0]
                            predictions.append(sentiment)
                            valid_reviews_for_df.append(review_text) # Add the original review
                
                if not predictions:
                    st.warning("After cleaning, no valid review text was found to analyze (e.g., all reviews were just numbers or punctuation).")
                else:
                    # --- 3. Aggregate & Display Results ---
                    sentiment_counts = pd.Series(predictions).value_counts()
                    
                    # Define our categories and their colors
                    categories_colors = {
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral": "#95a5a6"
                    }
                    
                    # Ensure all categories exist, even if count is 0
                    for category in categories_colors.keys():
                        if category not in sentiment_counts:
                            sentiment_counts[category] = 0
                    
                    # --- FINAL VERDICT ---
                    st.subheader("Final Verdict")
                    positive_count = sentiment_counts.get("Positive", 0)
                    negative_count = sentiment_counts.get("Negative", 0)
                    
                    if positive_count > (negative_count * 1.5): # Be more confident for "Happy"
                        st.success(f"**Overall: HAPPY** ({positive_count} positive vs {negative_count} negative)")
                        st.markdown("### ✅ **Consumers are generally happy.**")
                    elif negative_count > (positive_count * 1.5): # Be more confident for "Not Happy"
                        st.error(f"**Overall: NOT HAPPY** ({negative_count} negative vs {positive_count} positive)")
                        st.markdown("### ❌ **Consumers are generally not happy.**")
                    else:
                        st.warning(f"**Overall: MIXED / NEUTRAL** ({positive_count} positive, {negative_count} negative)")
                        st.markdown("### 🤷 **Consumer sentiment is mixed.**")

                    # --- Charts ---
                    st.subheader("Sentiment Breakdown")

                    # Create a new Series to guarantee the order for the chart
                    chart_data = pd.Series({
                        "Positive": sentiment_counts["Positive"],
                        "Negative": sentiment_counts["Negative"],
                        "Neutral": sentiment_counts["Neutral"]
                    })
                    
                    # Convert to a DataFrame and Transpose (T) it for st.bar_chart
                    chart_df = chart_data.to_frame().T
                    
                    # Get the colors in the *exact* order of the columns
                    chart_colors = [categories_colors[col] for col in chart_df.columns]

                    # Plot the DataFrame.
                    st.bar_chart(chart_df, color=chart_colors)

                    # --- Show the analyzed reviews ---
                    st.subheader("Analyzed Reviews")
                    
                    sample_df = pd.DataFrame({
                        'Entered Review': valid_reviews_for_df,
                        'Predicted Sentiment': predictions
                    })
                    st.dataframe(sample_df, use_container_width=True, height=300)

else:
    st.stop()
