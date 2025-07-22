from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def extract_pdf_content(file_path):
    """Extracts text from a PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_embeddings(text, model_path):
    """Creates embeddings using a local sentence transformer model."""
    model = SentenceTransformer(model_path)
    return model.encode(text.split("\n"))
