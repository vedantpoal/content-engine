import streamlit as st
from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStoreManager
from src.retrieval_engine import RetrievalEngine

def main():
    st.title("AI Content Engine - Document Comparison")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStoreManager()
    retrieval_engine = RetrievalEngine()
    
    # Process documents on first run
    if 'documents_processed' not in st.session_state:
        documents = doc_processor.process_documents('./documents')
        embeddings = embedding_generator.generate_embeddings(documents)
        vector_store.add_embeddings(documents, embeddings)
        st.session_state['documents_processed'] = True
    
    # Chat interface
    st.sidebar.header("Query Documents")
    query = st.sidebar.text_input("Enter your query:")
    
    if query:
        context = retrieval_engine.retrieve_context(
            query, 
            vector_store, 
            embedding_generator
        )
        
        response = retrieval_engine.generate_response(query, context)
        
        st.write("### Response")
        st.write(response)
        
        st.write("### Retrieved Context")
        for chunk in context:
            st.text_area("Context Chunk", chunk, height=100)

if __name__ == '__main__':
    main()