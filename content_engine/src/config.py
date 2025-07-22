import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File paths
PDF_FILES = {
    "alphabet": os.path.join(DATA_DIR, "alphabet_10k.pdf"),
    "tesla": os.path.join(DATA_DIR, "tesla_10k.pdf"),
    "uber": os.path.join(DATA_DIR, "uber_10k.pdf")
}
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index")
GPT4ALL_MODEL_FILE = os.path.join(MODELS_DIR, "gpt4all_model.bin")
