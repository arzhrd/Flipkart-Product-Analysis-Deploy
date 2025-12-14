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
        try:
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
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

    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.
        """
        # Encode
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
