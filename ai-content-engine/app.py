from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStoreManager
from src.retrieval_engine import RetrievalEngine

def initialize_content_engine(document_directory: str):
    """
    Initialize full content engine pipeline
    
    Args:
        document_directory (str): Path to PDF documents
    """
    # Document Processing
    doc_processor = DocumentProcessor()
    documents = doc_processor.process_documents(document_directory)
    
    # Embedding Generation
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(documents)
    
    # Vector Store Management
    vector_store = VectorStoreManager()
    vector_store.add_embeddings(documents, embeddings)
    
    return {
        'documents': documents,
        'embedding_generator': embedding_generator,
        'vector_store': vector_store
    }

def main():
    # Initialize content engine
    content_engine = initialize_content_engine('./documents')
    
    # Example query demonstration
    retrieval_engine = RetrievalEngine()
    query = "What are the primary risk factors for Google?"
    
    context = retrieval_engine.retrieve_context(
        query, 
        content_engine['vector_store'], 
        content_engine['embedding_generator']
    )
    
    response = retrieval_engine.generate_response(query, context)
    print(response)

if __name__ == '__main__':
    main()