# ==========================================
# FLIPKART REVIEW ANALYZER - STREAMLIT APP
# (Real Scraping Version)
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

# Scraping Libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Set up NLTK components
@st.cache_data
def load_nltk_data():
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
# 2. PREPROCESSING FUNCTION
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
# 3. REAL WEB SCRAPING FUNCTION (Using Selenium)
# ==========================================

@st.cache_data(show_spinner=False) # Cache the scraping result for a given URL
def get_real_reviews_from_flipkart(url, max_reviews=50):
    """
    Uses Selenium to open a real browser, find the 'All Reviews'
    page, and scrape the text.
    """
    
    # --- *** CRITICAL: UPDATE THESE SELECTORS *** ---
    # Flipkart changes its HTML classes often. These WILL break.
    # You must find the new ones using Chrome's "Inspect" tool.
    ALL_REVIEWS_LINK_XPATH = "//a[contains(., 'All Reviews') or contains(., 'All') and contains(., 'reviews')]"
    REVIEW_TEXT_CLASS = "ZmyHeo" # As of late 2025, this class holds the review text.
    # ---
    
    reviews = []
    
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # Run in the background
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = None
    try:
        with st.spinner(f"Starting browser and navigating to product page..."):
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            time.sleep(3) # Wait for page to load

        # Find and click the "All Reviews" link
        with st.spinner("Finding 'All Reviews' link..."):
            all_reviews_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, ALL_REVIEWS_LINK_XPATH))
            )
            driver.execute_script("arguments[0].click();", all_reviews_button)
            time.sleep(3) # Wait for review page to load

        with st.spinner(f"Scraping up to {max_reviews} reviews... (This may take a moment)"):
            last_height = driver.execute_script("return document.body.scrollHeight")
            
            while len(reviews) < max_reviews:
                # Parse the currently loaded HTML
                soup = BeautifulSoup(driver.page_source, "lxml")
                review_elements = soup.find_all("div", class_=REVIEW_TEXT_CLASS)
                
                new_reviews_found = 0
                for elem in review_elements:
                    review_text = elem.get_text(strip=True)
                    if review_text not in reviews:
                        reviews.append(review_text)
                        new_reviews_found += 1
                        if len(reviews) >= max_reviews:
                            break
                
                # If we're at the max or no new reviews were found on this scroll, stop
                if len(reviews) >= max_reviews or new_reviews_found == 0:
                    break
                
                # Scroll down to load more
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2) # Wait for new reviews to load
                
                # Check if we're at the bottom of the page
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    st.warning(f"Reached end of reviews. Found {len(reviews)} total.")
                    break
                last_height = new_height

    except Exception as e:
        st.error(f"Scraping Error: {e}")
        st.error("This often happens if Flipkart changed its HTML or blocked the request. The scraper's selectors may need to be updated.")
        return [] # Return empty list on failure
    finally:
        if driver:
            driver.quit() # Always close the browser

    return reviews[:max_reviews]


# ==========================================
# 4. STREAMLIT UI & MAIN LOGIC
# ==========================================

st.set_page_config(page_title="Flipkart Review Analyzer", page_icon="🛍️", layout="wide")
st.title("🛍️ Flipkart Product Review Analyzer")
st.markdown("Enter a Flipkart product URL to analyze customer sentiment and see if most consumers are satisfied.")

if model and vectorizer and le:
    url = st.text_input("Enter a Flipkart Product URL:", placeholder="https://www.flipkart.com/...")

    if st.button("Analyze Sentiment", type="primary"):
        if not url.startswith("https://www.flipkart.com/"):
            st.error("Please enter a valid Flipkart URL.")
        else:
            # --- 1. Scrape ---
            reviews_list = get_real_reviews_from_flipkart(url, max_reviews=50)
            
            if not reviews_list:
                st.warning("Could not find any reviews for this product. (Or scraping failed)")
            else:
                # --- 2. Process & Predict ---
                st.subheader(f"Analyzing {len(reviews_list)} real reviews...")
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
                    # --- 3. Aggregate & Display Results ---
                    sentiment_counts = pd.Series(predictions).value_counts()
                    categories_colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
                    for category in categories_colors.keys():
                        if category not in sentiment_counts:
                            sentiment_counts[category] = 0
                    
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
                    chart_data = pd.Series({
                        "Positive": sentiment_counts["Positive"],
                        "Negative": sentiment_counts["Negative"],
                        "Neutral": sentiment_counts["Neutral"]
                    })
                    chart_df = chart_data.to_frame().T
                    chart_colors = [categories_colors[col] for col in chart_df.columns]
                    st.bar_chart(chart_df, color=chart_colors)

                    # --- Show the REAL reviews ---
                    st.subheader(f"Scraped Reviews (Analyzed {len(predictions)} of {len(reviews_list)})")
                    sample_df = pd.DataFrame({
                        'Scraped Review Text': valid_reviews_for_df,
                        'Predicted Sentiment': predictions
                    })
                    st.dataframe(sample_df, use_container_width=True, height=300)

else:
    st.stop()
