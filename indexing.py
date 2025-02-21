import os
import glob
import streamlit as st
import requests
from types import SimpleNamespace
from document_loader import load_document

# Updated imports for llama-index 0.12.x
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Document, load_index_from_storage, StorageContext, Settings
from llama_index.core.llms import LLM, ChatMessage, MessageRole, CompletionResponse, CompletionResponseGen, ChatResponse
from typing import Any, Sequence, Optional, Dict, List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

class LMStudioLLM(LLM):
    def __init__(
        self,
        temperature: float = 0.1,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ):
        """Initialize LMStudioLLM."""
        super().__init__()
        self._base_url = "http://localhost:1234/v1"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        
        # Get available model
        response = requests.get(f"{self._base_url}/models")
        if response.status_code != 200:
            raise ValueError("Failed to get models from LM Studio")
        models = response.json().get("data", [])
        if not models:
            raise ValueError("No models available in LM Studio")
        self._model_id = models[0]['id']

    def complete(self, prompt: str, **kwargs: Any) -> ChatResponse:
        """Synchronous completion endpoint."""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model_id,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from LM Studio: {response.text}")
            
        response_data = response.json()
        return ChatResponse(message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_data['choices'][0]['message']['content']
        ))

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Synchronous chat endpoint."""
        formatted_messages = []
        if self._system_prompt:
            formatted_messages.append({"role": "system", "content": self._system_prompt})
        
        for message in messages:
            formatted_messages.append({
                "role": message.role.value,
                "content": message.content
            })
        
        response = requests.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model_id,
                "messages": formatted_messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from LM Studio: {response.text}")
            
        response_data = response.json()
        return ChatResponse(message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_data['choices'][0]['message']['content']
        ))

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Synchronous streaming completion endpoint."""
        raise NotImplementedError("stream_complete is not implemented.")

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        """Synchronous streaming chat endpoint."""
        raise NotImplementedError("stream_chat is not implemented.")

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Asynchronous completion endpoint."""
        raise NotImplementedError("acomplete is not implemented.")

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponse:
        """Asynchronous chat endpoint."""
        raise NotImplementedError("achat is not implemented.")

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Asynchronous streaming completion endpoint."""
        raise NotImplementedError("astream_complete is not implemented.")

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        """Asynchronous streaming chat endpoint."""
        raise NotImplementedError("astream_chat is not implemented.")

    @property
    def metadata(self) -> Any:
        """
        Get LLM metadata.
        
        Returns a SimpleNamespace object with all required fields for llama-index.
        """
        return SimpleNamespace(
            model_name=self._model_id,
            context_window=3900,  # Example value, adjust based on your model
            model_type="chat",
            num_output=512,  # Maximum number of tokens to generate
            is_chat_model=True,
            is_function_calling_model=False
        )

def configure_settings(llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Configures global settings for llama-index.

    Args:
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System prompt/instructions for the LLM.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=local_embedding_model_path,
        trust_remote_code=True,
        device="cuda"
    )
    llm = LMStudioLLM(
        temperature=0.1,
        max_tokens=256,
        system_prompt=system_instructions
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

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
    configure_settings(llm_model_path, local_embedding_model_path, system_instructions)
    index = VectorStoreIndex.from_documents(documents)
    return index

def persist_index(index, filepath):
    """
    Saves the index to disk.
    
    Args:
        index (VectorStoreIndex): The index to persist.
        filepath (str): The file path where the index should be saved.
    """
    index.storage_context.persist(persist_dir=filepath)

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
        # Create storage context from the specified filepath
        storage_context = StorageContext.from_defaults(persist_dir=filepath)
        
        # Load the index with the storage context
        configure_settings(llm_model_path, local_embedding_model_path, system_instructions)
        index = load_index_from_storage(
            storage_context=storage_context
        )
        
        # Store the storage context in session state for future use
        st.session_state.storage_context = storage_context
        
        return index
    except Exception as e:
        st.error(f"Failed to load index from disk: {e}")
        return None

def load_local_index(data_dir, llm_model_path, local_embedding_model_path, system_instructions=None):
    """
    Attempts to load a persistent index using the session's StorageContext.
    If unsuccessful, creates a new index.
    
    Args:
        data_dir (str): Directory containing the documents.
        llm_model_path (str): Path to your LLM model.
        local_embedding_model_path (str): Path to the local embedding model.
        system_instructions (str, optional): System instructions for the LLM.
    
    Returns:
        VectorStoreIndex: The loaded or newly created index.
    """
    try:
        documents = load_documents_from_dir(data_dir)
        configure_settings(llm_model_path, local_embedding_model_path, system_instructions)
        
        # Initialize storage context if it doesn't exist
        if 'storage_context' not in st.session_state:
            st.session_state.storage_context = StorageContext.from_defaults()
            
        # Create index with the storage context
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=st.session_state.storage_context
        )
        return index
        
    except FileNotFoundError as e:
        st.error(f"Error loading documents: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading index: {e}")
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
