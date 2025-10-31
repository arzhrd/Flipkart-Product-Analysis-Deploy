# ==========================================
# CUSTOMER SENTIMENT ANALYZER - STREAMLIT APP
# (Redesigned Version)
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
# 1. PAGE CONFIGURATION
# ==========================================

# Set page config first
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SETUP & MODEL LOADING
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
# 3. PREPROCESSING FUNCTION
# ==========================================

@st.cache_data
def clean(text):
    """
    The exact same cleaning pipeline from your notebook.
    """
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

# ==========================================
# 4. STYLED "VERDICT" BOX (Helper Function)
# ==========================================

def show_verdict(verdict_type, message, details):
    """Displays a custom-styled 'verdict' box."""
    
    # Define styles
    if verdict_type == "Happy":
        icon = "✅"
        border_color = "#2ecc71"
    elif verdict_type == "Not Happy":
        icon = "❌"
        border_color = "#e74c3c"
    else:
        icon = "🤷"
        border_color = "#95a5a6"
        
    # Use st.markdown with unsafe_allow_html to create a styled "card"
    st.markdown(
        f"""
        <div style="
            border-left: 10px solid {border_color};
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h3 style="margin-top: 0;">{icon} Overall: {verdict_type}</h3>
            <p style="font-size: 1.1em; margin-bottom: 0;">{message}</p>
            <p style="color: #555; margin-top: 10px; margin-bottom: 0;">{details}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 5. STREAMLIT UI & MAIN LOGIC
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.title("✨ About")
    st.info(
        """
        This app uses a Machine Learning model to analyze customer sentiment.
        
        **How to use:**
        1.  Paste in a list of reviews (one per line).
        2.  Click 'Analyze Sentiment'.
        3.  See the overall verdict and detailed breakdown.
        
        **Models Used:**
        * TF-IDF Vectorizer
        * Linear SVM (Sentiment Model)
        """
    )

# --- Main Page ---
st.title("🛍️ Customer Sentiment Analyzer")
st.markdown("Paste in customer reviews (one review per line) to instantly analyze overall sentiment.")

# Only proceed if models were loaded successfully
if model and vectorizer and le:
    
    review_input_placeholder = (
        "This product is amazing!\n"
        "Worst purchase ever. I'm returning it.\n"
        "It's okay, not great but not terrible.\n"
        "I love the color and the quality."
    )
    
    user_reviews = st.text_area(
        "Paste reviews here:",
        placeholder=review_input_placeholder,
        height=250,
        label_visibility="collapsed"
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        
        if not user_reviews.strip():
            st.error("Please enter at least one review.")
        else:
            reviews_list = [review.strip() for review in user_reviews.split('\n') if review.strip()]
            
            if not reviews_list:
                st.error("Please enter valid review text.")
            else:
                st.subheader(f"Analysis Results for {len(reviews_list)} Reviews")
                
                predictions = []
                valid_reviews_for_df = [] 
                
                with st.spinner("Running sentiment analysis model..."):
                    for review_text in reviews_list:
                        cleaned_review = clean(review_text)
                        
                        if cleaned_review:
                            vectorized_review = vectorizer.transform([cleaned_review])
                            prediction_int = model.predict(vectorized_review)[0]
                            sentiment = le.inverse_transform([prediction_int])[0]
                            predictions.append(sentiment)
                            valid_reviews_for_df.append(review_text) 
                
                if not predictions:
                    st.warning("After cleaning, no valid review text was found to analyze.")
                else:
                    sentiment_counts = pd.Series(predictions).value_counts()
                    categories_colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
                    
                    for category in categories_colors.keys():
                        if category not in sentiment_counts:
                            sentiment_counts[category] = 0
                    
                    # --- Create Columns for Dashboard ---
                    col1, col2 = st.columns([1.2, 1]) # Make first column slightly wider
                    
                    with col1:
                        # --- FINAL VERDICT (in left column) ---
                        st.markdown("#### Final Verdict")
                        positive_count = sentiment_counts.get("Positive", 0)
                        negative_count = sentiment_counts.get("Negative", 0)
                        details_text = f"({positive_count} positive vs {negative_count} negative)"
                        
                        if positive_count > (negative_count * 1.5):
                            show_verdict("Happy", "Consumers are generally happy.", details_text)
                        elif negative_count > (positive_count * 1.5):
                            show_verdict("Not Happy", "Consumers are generally not happy.", details_text)
                        else:
                            show_verdict("Mixed", "Consumer sentiment is mixed.", details_text)

                    with col2:
                        # --- CHARTS (in right column) ---
                        st.markdown("#### Sentiment Breakdown")
                        chart_data = pd.Series({
                            "Positive": sentiment_counts["Positive"],
                            "Negative": sentiment_counts["Negative"],
                            "Neutral": sentiment_counts["Neutral"]
                        })
                        chart_df = chart_data.to_frame().T
                        chart_colors = [categories_colors[col] for col in chart_df.columns]
                        st.bar_chart(chart_df, color=chart_colors)

                    # --- Show the analyzed reviews in an expander ---
                    st.divider() # Adds a horizontal line
                    with st.expander("Show Analyzed Reviews"):
                        sample_df = pd.DataFrame({
                            'Entered Review': valid_reviews_for_df,
                            'Predicted Sentiment': predictions
                        })
                        st.dataframe(sample_df, use_container_width=True, height=300)

else:
    # This runs if the models failed to load
    st.error("Application failed to load. Please check model files.")
    st.stop()
