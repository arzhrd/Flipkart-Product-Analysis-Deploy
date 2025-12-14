import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from vector_db import VectorStore

def test_persistence():
    print("Testing VectorStore Persistence...")
    
    # Setup
    vs = VectorStore("test_persist.bin", "test_persist.pkl")
    
    # Create fake embeddings
    embeddings = np.random.rand(5, 384).astype('float32') # 5 vectors, 384 dim (same as MiniLM)
    metadata = [f"Item {i}" for i in range(5)]
    
    # Build
    print("Building index...")
    vs.build_index(embeddings, metadata)
    assert vs.index.ntotal == 5
    
    # Save
    print("Saving index...")
    vs.save_index()
    assert os.path.exists("test_persist.bin")
    
    # Reset
    print("Clearing memory...")
    vs = None
    
    # Load
    print("Loading index...")
    vs_new = VectorStore("test_persist.bin", "test_persist.pkl")
    success = vs_new.load_index()
    
    assert success, "Load failed"
    assert vs_new.index.ntotal == 5, "Index count mismatch after load"
    assert len(vs_new.metadata) == 5, "Metadata count mismatch"
    
    print("Searching new index...")
    # Search
    d, i, r = vs_new.search(embeddings[0], k=1)
    print(f"Result: {r[0]}")
    assert r[0] == "Item 0", "Search failed consistency check"
    
    print("PERSISTENCE TEST PASSED")
    
    # Cleanup
    if os.path.exists("test_persist.bin"):
        os.remove("test_persist.bin")
    if os.path.exists("test_persist.pkl"):
        os.remove("test_persist.pkl")

if __name__ == "__main__":
    test_persistence()
