import streamlit as st
from utils.helpers import extract_pdf_content, create_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from chromadb import Client
from chromadb.utils import embedding_functions

# Load models and vector store
llm_model_path = "models/local_llm"
st_model_path = "models/sentence_transformer"
chromadb_client = Client.from_local_settings("./chromadb_store")

tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)

# Streamlit UI
st.title("Content Engine")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Extract text and create embeddings
    text = extract_pdf_content(uploaded_file)
    embeddings = create_embeddings(text, st_model_path)

    # Add data to ChromaDB
    collection = chromadb_client.get_or_create_collection("pdf_data")
    collection.add(embeddings=embeddings, metadatas=[{"text": line} for line in text.split("\n")])

    st.success("PDF content added to the vector store!")

    # Query interface
    query = st.text_input("Enter your query:")
    if query:
        query_embedding = create_embeddings(query, st_model_path)
        results = collection.query(embedding=query_embedding, n_results=3)
        st.write("Results:", results)
