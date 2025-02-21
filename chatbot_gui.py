import streamlit as st
import os
import shutil
from indexing import create_index, load_local_index, query_index, persist_index
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from session_manager import load_chat_history, save_chat_history, clear_chat_history

# --- Configuration ---
INDEX_FILEPATH = "vector_index.json"
DATA_DIR = "data"
MODEL_PATH = st.text_input("LM Studio API Endpoint", value="http://localhost:1234", help="Enter the IP address and port where LM Studio is running.")  # Replace with your model path
LOCAL_EMBEDDING_MODEL_PATH = "embedding_model\\snowflake-arctic-embed-m-v2.0"
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- Sidebar for settings and instructions ---
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
    system_instructions = st.text_area("System Instructions", value="You are a helpful chatbot that answers questions about the documents provided.")
    clear_data_button = st.button("Clear Data Directory")
    clear_index_button = st.button("Clear Index")
    clear_history_button = st.button("Clear Chat History")

    if clear_data_button:
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)
        st.success("Data directory cleared!")
    if clear_index_button:
        if os.path.exists(INDEX_FILEPATH):
            os.remove(INDEX_FILEPATH)
        st.session_state.index = None  # Clear the index from session state
        st.success("Index cleared!")
    if clear_history_button:
        if st.session_state.current_session:
            clear_chat_history(st.session_state.current_session)
            st.session_state.chat_history = []
        st.success("Chat history cleared!")

# --- Initialize session state ---
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = "default_session" # Or generate a unique session ID
if 'system_instructions' not in st.session_state:
    st.session_state.system_instructions = system_instructions # Load default from text_area, or set a default string

# --- Load chat history from session ---
if st.session_state.current_session:
    st.session_state.chat_history = load_chat_history(st.session_state.current_session)

# --- Main UI ---
st.title("RAG Chatbot with Local LM Studio")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Document upload handling
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Documents uploaded successfully!")

    # Create or load index
    if not os.path.exists(INDEX_FILEPATH) or st.session_state.index is None:
        with st.spinner('Creating index...'):
            try:
                st.session_state.index = create_index(DATA_DIR, MODEL_PATH, LOCAL_EMBEDDING_MODEL_PATH, system_instructions=st.session_state.system_instructions)
                persist_index(st.session_state.index, INDEX_FILEPATH)
                st.session_state.chat_history.append({"role": "assistant", "content": "Index created and saved."})
            except ValueError as e:
                st.error(f"Error creating index: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error creating index: {e}"})
            except Exception as e:
                st.error(f"Unexpected error during index creation: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Unexpected error during index creation: {e}"})
    else:
        try:
            with st.spinner('Loading existing index...'):
                st.session_state.index = load_local_index(DATA_DIR, MODEL_PATH, LOCAL_EMBEDDING_MODEL_PATH, system_instructions=st.session_state.system_instructions) #load_index_from_disk(INDEX_FILEPATH, MODEL_PATH, EMBEDDING_MODEL_NAME, system_instructions=st.session_state.system_instructions)
                if st.session_state.index is None:
                    st.session_state.index = create_index(DATA_DIR, MODEL_PATH, LOCAL_EMBEDDING_MODEL_PATH, system_instructions=st.session_state.system_instructions)
                    persist_index(st.session_state.index, INDEX_FILEPATH)
                st.session_state.chat_history.append({"role": "assistant", "content": "Index loaded from disk."})
        except Exception as e:
            st.error(f"Error loading index: {e}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error loading index: {e}"})

# Chat input
if prompt := st.chat_input("Ask questions about your documents"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is not None:
        with st.spinner('Thinking...'):
            system_prompt = st.session_state.system_instructions
            response = query_index(st.session_state.index, prompt)
            # Add AI message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning("Please upload documents to create an index first.")
        st.session_state.chat_history.append({"role": "assistant", "content": "Please upload documents to create an index first."})
        with st.chat_message("assistant"):
            st.markdown("Please upload documents to create an index first.")
            st.markdown("Please upload documents to create an index first.")

    # Save the chat history
    save_chat_history(st.session_state.current_session, st.session_state.chat_history)
