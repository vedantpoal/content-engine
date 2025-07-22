from config import FAISS_INDEX_FILE, EMBEDDINGS_FILE, GPT4ALL_MODEL_FILE
from pygpt4all import GPT4All
import faiss
import numpy as np

def query_faiss(query_embedding, faiss_index, k=3):
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return indices[0]

def generate_response(llm, context):
    prompt = f"Given this context:\n{context}\nAnswer the query:"
    return llm.generate(prompt)

# Example usage:
if __name__ == "__main__":
    llm = GPT4All(GPT4ALL_MODEL_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)

    query_embedding = embeddings[0]  # Use a sample embedding for testing
    indices = query_faiss(query_embedding, faiss_index)
    context = " ".join([str(embeddings[i]) for i in indices])
    response = generate_response(llm, context)
    print(f"Response: {response}")
