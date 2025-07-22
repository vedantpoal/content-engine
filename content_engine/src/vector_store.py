from config import EMBEDDINGS_FILE, FAISS_INDEX_FILE
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_FILE)

# Example usage:
if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_FILE)
    create_faiss_index(embeddings)
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")
