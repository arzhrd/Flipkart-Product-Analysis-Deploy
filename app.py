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
        border_color = "#2ecc71"
    elif verdict_type == "Not Happy":
        icon = "❌"
        border_color = "#e74c3c"
    else:
        icon = "🤷"
        border_color = "#95a5a6"
        
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
                    
                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                         if pos > (neg * 1.5):
                            show_verdict("Happy", "Consumers are generally happy.", f"{pos} vs {neg}")
                         elif neg > (pos * 1.5):
                            show_verdict("Not Happy", "Consumers are generally unhappy.", f"{pos} vs {neg}")
                         else:
                            show_verdict("Mixed", "Consumer sentiment is mixed.", f"{pos} vs {neg}")
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
    
    # Upload Section
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
                    embeddings = emb_model.generate_embeddings(processed_texts)
                    
                    st.session_state.vector_store.build_index(embeddings, processed_texts)
                    st.session_state.vector_store.save_index()
                    st.success(f"Indexed {len(processed_texts)} reviews successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

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
