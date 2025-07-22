import chromadb
from typing import List, Dict
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self, persist_directory: str = './chroma_db'):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory (str): Directory to persist vector store
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.collection_name = 'document_collection'

    def create_collection(self, name: str = None):
        """
        Create or get an existing vector store collection
        
        Args:
            name (str, optional): Name of collection
        """
        # Use the provided name or default to the instance's collection name
        collection_name = name or self.collection_name
        
        try:
            # Try to get existing collection, or create if it doesn't exist
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"Error creating/getting collection: {e}")
            raise

    def add_embeddings(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ):
        """
        Add embeddings to vector store
        
        Args:
            documents (List[Document]): Source documents
            embeddings (List[List[float]]): Generated embeddings
        """
        # Ensure collection exists before adding
        if not self.collection:
            self.create_collection()

        # Prepare data for insertion
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=[f'doc_{i}' for i in range(len(documents))]
            )
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            raise

    def query(self, query_embedding: List[float], top_k: int = 5) -> Dict:
        """
        Query vector store
        
        Args:
            query_embedding (List[float]): Query vector
            top_k (int): Number of results
        
        Returns:
            Query results
        """
        if not self.collection:
            raise ValueError("Collection has not been created. Call create_collection() first.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )