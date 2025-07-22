import streamlit as st
from query_engine import query_faiss, generate_response
from vector_store import load_faiss_index
from embedding_generator import generate_embeddings
from config import EMBEDDINGS_FILE, GPT4ALL_MODEL_FILE

import numpy as np

# Load components
faiss_index = load_faiss_index()
embeddings = np.load(EMBEDDINGS_FILE)

st.title("Content Engine: PDF Comparison")
query = st.text_input("Enter your query:")

if query:
    # Generate query embedding
    query_embedding = generate_embeddings([query])[0]

    # Query FAISS index
    indices = query_faiss(query_embedding, faiss_index)
    context = " ".join([str(embeddings[i]) for i in indices])

    # Generate response
    from pygpt4all import GPT4All
    llm = GPT4All(GPT4ALL_MODEL_FILE)
    response = generate_response(llm, context)

    st.write(f"Response: {response}")
