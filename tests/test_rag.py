import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from etl_pipeline import DatasetLoader, TextPreprocessor, Embedder
from vector_db import VectorStore
from rag_engine import RAGPipeline, MockLLM

def test_pipeline():
    print("Testing Pipeline...")
    
    # 1. Load Data
    loader = DatasetLoader("sample_reviews.csv")
    df = loader.load()
    assert df is not None, "Data loading failed"
    print("Data loaded.")
    
    # 2. Preprocess
    texts = df['Review'].tolist()
    processed = [TextPreprocessor.clean_for_rag(t) for t in texts]
    assert len(processed) > 0, "Preprocessing failed"
    print("Data processed.")
    
    # 3. Embed
    embedder = Embedder() # This might download the model
    embeddings = embedder.generate_embeddings(processed)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 4. Index
    vector_store = VectorStore("test_index.bin", "test_metadata.pkl")
    vector_store.build_index(embeddings, processed)
    vector_store.save_index()
    
    # 5. Search
    query = "battery"
    q_emb = embedder.generate_embeddings([query])[0]
    _, _, results = vector_store.search(q_emb, k=1)
    print(f"Search for 'battery' found: {results[0]}")
    assert "battery" in results[0].lower(), "Search failed relevance check"
    
    # 6. RAG
    llm = MockLLM()
    rag = RAGPipeline(vector_store, llm)
    answer, context = rag.answer_query(query, embedder)
    print(f"RAG Answer: {answer}")
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_pipeline()
