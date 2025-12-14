import faiss
import numpy as np
import os
import pickle

class VectorStore:
    """Manages the FAISS vector database."""
    def __init__(self, index_path="faiss_index.bin", metadata_path="faiss_metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = [] # List to store corresponding text/metadata for each vector

    def build_index(self, embeddings, metadata):
        """
        Builds a FAISS index from embeddings.
        embeddings: numpy array of shape (n_samples, embedding_dim)
        metadata: list of strings (or dicts) corresponding to each embedding
        """
        if len(embeddings) == 0:
            print("No embeddings to index.")
            return

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = metadata
        print(f"Index built with {self.index.ntotal} vectors.")

    def save_index(self):
        """Saves the index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print("Index and metadata saved.")
        else:
            print("No index to save.")

    def load_index(self):
        """Loads index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print("Index and metadata loaded.")
            return True
        return False

    def search(self, query_embedding, k=3):
        """
        Searches the index for query_embedding.
        Returns: distances, indices, valid_metadata (results)
        """
        if not self.index:
            print("Index not loaded.")
            return [], [], []

        # faiss expects float32
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata) and idx >= 0:
                results.append(self.metadata[idx])
            else:
                results.append(None)
                
        return distances, indices, results
