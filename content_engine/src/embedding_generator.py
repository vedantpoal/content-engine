from config import EMBEDDINGS_FILE
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

# Example usage:
if __name__ == "__main__":
    sample_texts = [
        "Alphabet Inc. risk factors.",
        "Tesla, Inc. financial results.",
        "Uber Technologies business overview."
    ]
    embeddings = generate_embeddings(sample_texts)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
