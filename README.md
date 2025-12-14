# 🛍️ Flipkart Product Analysis AI (Sentiment + RAG)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![RAG](https://img.shields.io/badge/GenAI-RAG-green)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)](https://github.com/facebookresearch/faiss)

An advanced AI application that combines **Sentiment Analysis** with **Retrieval-Augmented Generation (RAG)** to provide deep insights into Flipkart product reviews.

**Core Capabilities:**
1.  **Sentiment Analyzer:** Instantly classifies reviews as Positive, Negative, or Neutral using a Linear SVM model.
2.  **Chat with Data (RAG):** Allows users to **ask questions** about product reviews (e.g., "Why are customers unhappy with the battery?") and get AI-generated answers based on actual customer feedback.

---

## ✨ Key Features

### 1. Sentiment Analysis Engine
*   **Real-Time Classification:** Validates 200,000+ review patterns to predict sentiment.
*   **Visual Dashboard:** Bar charts and "Verdict" cards (Happy/Not Happy).
*   **Preprocessing:** Custom NLTK pipeline for cleaning noise (HTML, URLs, Stopwords).

### 2. RAG Chatbot (New!) 🤖
*   **Talk to your Data:** Upload a CSV of reviews and ask questions in natural language.
*   **Context-Aware:** The AI retrieves specific reviews relevant to your query to generate an accurate answer.
*   **ETL Pipeline:** Automatically ingests, cleans, and chunks reviews.
*   **Vector Search:** Uses **FAISS** for millisecond-latency similarity search.

---

## 🛠️ Technical Stack

*   **Frontend:** Streamlit
*   **Sentiment Model:** Scikit-learn (Linear SVM), TF-IDF
*   **RAG Pipeline:**
    *   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
    *   **Vector DB:** FAISS (Facebook AI Similarity Search)
    *   **LLM Integration:** Google Gemini API (or Mock LLM for testing)
*   **Orchestration:** Python (Custom ETL & RAG classes)

---

## ⚙️ How It Works

### The RAG Pipeline (Chat with Data)
1.  **Ingest:** User uploads a CSV file.
2.  **Embed:** The `Embedder` class converts reviews into 384-dimensional vectors.
3.  **Index:** Vectors are stored in a **FAISS** index for efficient retrieval.
4.  **Retrieve:** When you ask a question, the system finds the top 3 most relevant reviews.
5.  **Generate:** A Generative AI model (Gemini) answers your question using those reviews as context.

---

## 🚦 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/arzhrd/Flipkart-Product-Analysis-Deploy.git
cd Flipkart-Product-Analysis-Deploy
```

### 2. Install Dependencies
```bash
# Recommended: Create a virtual environment first
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Using the App
*   **Sentiment Tab:** Paste text reviews to see if they are Positive or Negative.
*   **Chat with Data Tab:**
    1.  Upload a CSV file (e.g., the included `sample_reviews.csv`).
    2.  Click **"Process & Build Index"**.
    3.  (Optional) Enter a **Gemini API Key** in the sidebar for better answers.
    4.  Ask questions like *"What is the build quality like?"*

---

## 📁 File Structure

```
.
├── src/                        # <--- NEW: Core Logic
│   ├── etl_pipeline.py         # Data loading & cleaning
│   ├── vector_db.py            # FAISS index management
│   └── rag_engine.py           # RAG orchestration & LLM wrapper
├── tests/                      # Validation scripts
├── app.py                      # Main Streamlit Dashboard
├── requirements.txt            # Project dependencies
├── sentiment_model.pkl         # Pre-trained Sentiment Model
├── tfidf_vectorizer.pkl        # Pre-trained Vectorizer
└── sample_reviews.csv          # Demo data
```
