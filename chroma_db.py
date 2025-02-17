import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index import StorageContext

def initialize_chroma():
    """Initialize Chroma client, vector store, and storage context."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="rag_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return chroma_client, vector_store, storage_context
