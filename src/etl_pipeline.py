import pandas as pd
import re
import string
from sentence_transformers import SentenceTransformer
import streamlit as st

class DatasetLoader:
    """Loads dataset from a CSV file."""
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        """Loads and returns the dataframe."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                df = pd.read_csv(self.filepath, encoding=enc)
                print(f"Successfully loaded with {enc} encoding.")
                return df
            except Exception as e:
                print(f"Failed to load with {enc}: {e}")
                continue
        
        print("Failed to load dataset with supported encodings.")
        return None

class TextPreprocessor:
    """
    Preprocessing for RAG specifically. 
    We want to keep the text natural but clean up artifacts.
    """
    @staticmethod
    def clean_for_rag(text):
        if not isinstance(text, str):
            return ""
        
        # Lowercase mostly helps, but some models prefer case. 
        # sentence-transformers/all-MiniLM-L6-v2 is uncased mostly or handles it well.
        # We will keep it relatively raw but clean HTML and weird symbols.
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https.?:/\/\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', ' ', text) # Replace newlines with space
        # We DO NOT remove stop words or stem for RAG, as context flows better without it.
        return text.strip()

class Embedder:
    """Generates embeddings using SentenceTransformers."""
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts, batch_size=32, progress_callback=None):
        """
        Generates embeddings for a list of texts with batching and progress callback.
        """
        import numpy as np
        
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            # Encode batch
            batch_emb = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
            # Update progress
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])
