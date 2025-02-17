import os
import glob
import streamlit as st
from document_loader import load_document

# Updated imports for llama-index 0.12.x
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Document, ServiceContext, load_index_from_storage, StorageContext
from llama_index.llms.llama_cpp import LlamaCPP
from sentence_transformers import SentenceTransformer

# Rest of your indexing.py code remains the same
def load_documents_from_dir(data_dir):
    """
    Loads documents from the specified directory and returns a list of Document objects.
    """
    documents = []
    for filename in glob.glob(os.path.join(data_dir, '*')):
        text_content = load_document(filename)
        if text_content:
            documents.append(Document(text=text_content, doc_id=filename))
    if not documents:
        raise ValueError(f"No documents loaded from directory: {data_dir}")
    return documents

def build_service_context(llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Builds and returns a ServiceContext for llama-index.

    Args:
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System prompt/instructions for the LLM.

    Returns:
        ServiceContext: The configured service context.
    """
    embed_model = SentenceTransformer(local_embedding_model_path)
    llm = LlamaCPP(
        model_url="http://localhost:1234",
        temperature=0.1,
        max_new_tokens=256,
        verbose=False,
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    return service_context

def create_index(data_dir, llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Creates a VectorStoreIndex from documents in the specified directory.
    
    Args:
        data_dir (str): Directory containing the documents.
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System instructions for the LLM.
    
    Returns:
        VectorStoreIndex: The created index.
    """
    documents = load_documents_from_dir(data_dir)
    service_context = build_service_context(llm_model_path, local_embedding_model_path, system_instructions)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def persist_index(index, filepath):
    """
    Saves the index to disk.
    
    Args:
        index (VectorStoreIndex): The index to persist.
        filepath (str): The file path where the index should be saved.
    """
    index.save_to_disk(filepath)

def load_index_from_disk(filepath, llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Loads the index from disk using the provided StorageContext.
    
    Args:
        filepath (str): Path to the saved index file.
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System instructions for the LLM.
    
    Returns:
        VectorStoreIndex: The loaded index, or None if not found.
    """
    try:
        storage_context = StorageContext.from_defaults()
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        print(f"Failed to load index from disk: {e}")
        return None

def load_local_index(data_dir, llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Attempts to load a persistent index using the session's StorageContext.
    If unsuccessful, returns None.
    
    Args:
        data_dir (str): Directory containing the documents.
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System instructions for the LLM.
    
    Returns:
        VectorStoreIndex or None: The loaded index if available.
    """
    documents = load_documents_from_dir(data_dir)
    service_context = build_service_context(llm_model_path, local_embedding_model_path, system_instructions)
    try:
        if 'storage_context' in st.session_state:
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                storage_context=st.session_state.storage_context
            )
            return index
    except FileNotFoundError:
        return None

def query_index(index, query):
    """
    Queries the VectorStoreIndex with the provided query and returns the response.
    
    Args:
        index (VectorStoreIndex): The index to query.
        query (str): The query string.
    
    Returns:
        str: The response from the LLM.
    """
    if index is None:
        return "Index was not created. Please upload documents and create an index."
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)
