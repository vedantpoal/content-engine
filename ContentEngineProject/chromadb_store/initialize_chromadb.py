from chromadb.config import Settings
from chromadb import Client

# Initialize ChromaDB client
client = Client(Settings(persist_directory="./chromadb_store", chroma_db_impl="duckdb+parquet"))
print("ChromaDB initialized successfully!")
