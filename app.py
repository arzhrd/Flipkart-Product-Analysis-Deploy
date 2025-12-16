# ==========================================
# CUSTOMER SENTIMENT ANALYZER & RAG BOT
# ==========================================
import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from etl_pipeline import DatasetLoader, TextPreprocessor, Embedder
    from vector_db import VectorStore
    from rag_engine import RAGPipeline, GeminiLLM, MockLLM
except ImportError as e:
    st.error(f"Error importing RAG modules: {e}")

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Flipkart Review Analyzer(RAG)",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - FUTURISTIC THEME
# ==========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;700&display=swap');

    :root {
        --primary-color: #00f2ea; /* Neon Cyan */
        --secondary-color: #ff0055; /* Neon Pink/Red */
        --bg-color: #0b0c10;
        --card-bg: rgba(31, 40, 51, 0.7);
        --text-color: #c5c6c7;
        --heading-font: 'Orbitron', sans-serif;
        --body-font: 'Roboto', sans-serif;
    }

    /* Global Styles */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 242, 234, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(255, 0, 85, 0.1) 0%, transparent 20%);
        font-family: var(--body-font);
        color: var(--text-color);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--heading-font);
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 242, 234, 0.3);
    }

    /* Sidebar - Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 12, 16, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 242, 234, 0.2);
    }
    
    div[data-testid="stSidebarNav"] {
        background-image: none;
    }

    /* Buttons */
    .stButton > button {
        background: transparent;
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
        border-radius: 5px;
        font-family: var(--heading-font);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background: var(--primary-color);
        color: #000;
        box-shadow: 0 0 15px var(--primary-color);
        border-color: var(--primary-color);
    }
    
    /* Primary Button (Use specific selector if cleaner, but logic remains same) */
    /* Streamlit's primary button has different class usually, but global override works well for theme */

    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: #fff;
        border: 1px solid rgba(0, 242, 234, 0.3);
        border-radius: 5px;
        font-family: var(--body-font);
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 10px rgba(0, 242, 234, 0.2);
    }

    /* Cards / Expanders */
    .streamlit-expanderHeader {
        background-color: var(--card-bg);
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: var(--heading-font);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid rgba(255,255,255,0.2);
        color: #aaa;
        border-radius: 5px;
        font-family: var(--heading-font);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(0, 242, 234, 0.1);
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
    }

    /* Verdict Card Custom Class */
    .verdict-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .verdict-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--border-color);
        box-shadow: 0 0 10px var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETUP & MODEL LOADING (Sentiment)
# ==========================================

@st.cache_data
def load_nltk_data():
    """Downloads NLTK data and returns stopwords and stemmer."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        return set(stopwords.words('english')), SnowballStemmer("english")
    except:
        return set(), SnowballStemmer("english")

stopword, stemmer = load_nltk_data()

@st.cache_resource
def load_sentiment_models():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, vectorizer, le
    except:
        return None, None, None

sentiment_model, sentiment_vectorizer, sentiment_le = load_sentiment_models()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

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

def show_verdict(verdict_type, message, details):
    if verdict_type == "Happy":
        icon = "✅"
        color = "#00f2ea" # Neon Cyan
    elif verdict_type == "Not Happy":
        icon = "❌"
        color = "#ff0055" # Neon Red
    else:
        icon = "🤷"
        color = "#f39c12" # Neon Orange/Yellow vibe

    st.markdown(
        f"""
        <div class="verdict-card" style="--border-color: {color};">
            <h3 style="margin-top: 0; color: {color}; text-shadow: 0 0 5px {color};">{icon} Overall: {verdict_type}</h3>
            <p style="font-size: 1.1em; margin-bottom: 0;">{message}</p>
            <p style="color: #888; margin-top: 10px; margin-bottom: 0; font-family: 'Roboto Mono', monospace;">{details}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 4. INITIALIZE RAG STATE
# ==========================================

# Define absolute paths for persistence
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index.bin')
DB_META_PATH = os.path.join(os.path.dirname(__file__), 'faiss_metadata.pkl')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore(index_path=DB_FAISS_PATH, metadata_path=DB_META_PATH)
    if st.session_state.vector_store.load_index():
        st.toast("Index loaded from disk!", icon="✅")
    else:
        st.toast("No existing index found.", icon="ℹ️")

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'embedder' not in st.session_state:
    # Initialize lazily to avoid heavy load on startup if not needed
    st.session_state.embedder = None 

def get_embedder():
    if st.session_state.embedder is None:
        with st.spinner("Loading Embedding Model..."):
            st.session_state.embedder = Embedder()
    return st.session_state.embedder

# ==========================================
# 5. MAIN UI
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.title("🛍️ Flipkart AI")
    
    with st.expander("ℹ️ About the Project", expanded=True):
        st.write("""
        This project has been upgraded with **GenAI capabilities**!
        
        **New Features:**
        - **ETL Pipeline**: Automatically cleans and processes raw review data.
        - **Vector DB (FAISS)**: Stores reviews as embeddings for fast search.
        - **RAG Engine**: Retrieves relevant reviews to answer your questions using an LLM.
        """)
        
    with st.expander("🛠️ How it Works"):
        st.write("""
        1. **Upload**: You upload a CSV of reviews.
        2. **Index**: The app chunks text and creates a FAISS index.
        3. **Ask**: You ask a question (e.g., "Is the battery good?").
        4. **Retrieve**: The app finds the top 3 related reviews.
        5. **Answer**: Gemini (or Mock LLM) generates an answer based *only* on those reviews.
        """)
    
    st.divider()
    st.markdown("### 🔑 API Config")
    api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Leave empty to use Mock LLM.")
    
    st.divider()
    st.info("Built with Streamlit, LangChain concepts, and FAISS.")

st.title("🛍️ Flipkart Product Analysis AI")

tab1, tab2 = st.tabs(["🙂 Sentiment Analysis", "🤖 Chat with Data"])

# --- TAB 1: SENTIMENT ---
with tab1:
    st.header("Customer Customer Sentiment")
    
    st.markdown("[![View Code on GitHub](https://img.shields.io/badge/View%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/arzhrd/Flipkart-Product-Analysis-Deploy)")
    st.link_button("View Code on GitHub", "https://github.com/arzhrd/Flipkart-Product-Analysis-Deploy")
    
    if sentiment_model and sentiment_vectorizer and sentiment_le:
        user_reviews = st.text_area("Paste reviews here:", height=200)
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_reviews.strip():
                reviews_list = [r.strip() for r in user_reviews.split('\n') if r.strip()]
                
                predictions = []
                valid_reviews = []
                
                for review in reviews_list:
                    cleaned = clean_sentiment(review)
                    if cleaned:
                        vec = sentiment_vectorizer.transform([cleaned])
                        pred = sentiment_model.predict(vec)[0]
                        label = sentiment_le.inverse_transform([pred])[0]
                        predictions.append(label)
                        valid_reviews.append(review)
                
                if predictions:
                    counts = pd.Series(predictions).value_counts()
                    pos = counts.get("Positive", 0)
                    neg = counts.get("Negative", 0)
                    neu = counts.get("Neutral", 0)
                    
                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                         if pos > (neg * 1.5):
                            show_verdict("Happy", "Consumers are generally happy.", f"{pos} Pos vs {neg} Neg ({neu} Neu)")
                         elif neg > (pos * 1.5):
                            show_verdict("Not Happy", "Consumers are generally unhappy.", f"{pos} Pos vs {neg} Neg ({neu} Neu)")
                         else:
                            show_verdict("Mixed", "Consumer sentiment is mixed.", f"{pos} Pos vs {neg} Neg ({neu} Neu)")
                    with c2:
                         st.bar_chart(counts)
                    
                    with st.expander("Detailed Results"):
                        st.dataframe(pd.DataFrame({"Review": valid_reviews, "Sentiment": predictions}))
            else:
                st.warning("Please enter some text.")
    else:
        st.error("Sentiment models not found.")

# --- TAB 2: RAG ---
with tab2:
    st.header("Chat with Reviews (RAG)")

    st.warning("⚠️ **Note:** This app runs on Streamlit Community Cloud, which has resource limits. Please upload smaller CSV files for best results. For analyzing large datasets, run this app locally.")
    
    st.markdown("[![View Code on GitHub](https://img.shields.io/badge/View%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/arzhrd/Flipkart-Product-Analysis-Deploy)")
    st.link_button("View Code on GitHub", "https://github.com/arzhrd/Flipkart-Product-Analysis-Deploy")

    with st.expander("📂 Upload & Process Data", expanded=False):
        uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])
        if uploaded_file and st.button("Process & Build Index"):
            try:
                # Save temp
                with open("temp_reviews.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = DatasetLoader("temp_reviews.csv")
                df = loader.load()
                
                if df is not None:
                    # Assume column is 'Review' or 'review' or the first text column
                    text_col = None
                    for col in df.columns:
                        if 'review' in col.lower() or 'text' in col.lower():
                            text_col = col
                            break
                    if not text_col:
                        text_col = df.columns[0] # Fallback
                    
                    st.write(f"Using column: `{text_col}`")
                    
                    # Preprocess
                    raw_texts = df[text_col].dropna().astype(str).tolist()
                    processed_texts = [TextPreprocessor.clean_for_rag(t) for t in raw_texts if len(str(t)) > 5]
                    
                    # Embed & Index
                    emb_model = get_embedder()
                    
                    # Progress Bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    def update_progress(current, total):
                        frac = current / total
                        progress_bar.progress(frac)
                        elapsed = time.time() - start_time
                        if frac > 0:
                            rate = elapsed / frac
                            remaining = rate - elapsed
                            status_text.caption(f"⏳ Processing: {int(frac*100)}% | Est. Time Left: {remaining:.1f}s")

                    embeddings = emb_model.generate_embeddings(processed_texts, progress_callback=update_progress)
                    
                    # Cleanup
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.session_state.vector_store.build_index(embeddings, processed_texts)
                    st.session_state.vector_store.save_index()
                    st.success(f"Indexed {len(processed_texts)} reviews successfully!")
            except Exception as e:
                import traceback
                st.error(f"Error processing file: {e}\n\n{traceback.format_exc()}")

    # Chat Section
    st.divider()
    query = st.text_input("Ask a question about the products:")
    
    if st.button("Ask AI"):
        if not query:
            st.warning("Please ask a question.")
        else:
            # Check if index exists, if not, try to load from disk
            if not st.session_state.vector_store.index:
                st.session_state.vector_store.load_index()

            if not st.session_state.vector_store.index:
                st.error("Index is empty. Please upload and process data first.")
            else:
                emb_model = get_embedder()
                
                # Setup LLM
                if api_key:
                    llm = GeminiLLM(api_key=api_key)
                else:
                    llm = MockLLM()
                    st.info("Using Mock LLM (No API Key provided).")
                
                rag = RAGPipeline(st.session_state.vector_store, llm)
                
                with st.spinner("Thinking..."):
                    answer, context = rag.answer_query(query, emb_model)
                
                st.markdown(f"**Answer:**\n\n{answer}")
                
                with st.expander("View Source Context"):
                    for i, ctx in enumerate(context):
                        st.markdown(f"**Source {i+1}:** {ctx}")
