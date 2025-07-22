from typing import List
from sentence_transformers import SentenceTransformer
import torch
from langchain.docstore.document import Document

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_name (str): Hugging Face model name
        """
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    
    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for document chunks
        
        Args:
            documents (List[Document]): List of document chunks
        
        Returns:
            List of embeddings
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()