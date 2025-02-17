import streamlit as st
import shutil
import os

DATA_DIRECTORY = "data"  # Directory for uploaded documents

def clear_data_directory(directory):
    """Clear all files and subdirectories in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def new_chat_session():
    """Start a new chat session by clearing previous data and state."""
    clear_data_directory(DATA_DIRECTORY)
    st.session_state.chat_history = []
    st.session_state.index = None
    st.session_state.session_name = f"Chat {len(st.session_state.sessions) + 1}"
    st.session_state.sessions.append(st.session_state.session_name)
    st.session_state.current_session = st.session_state.session_name
    # Clear Chroma DB related state
    for key in ['chroma_client', 'vector_store', 'storage_context']:
        if key in st.session_state:
            del st.session_state[key]

def load_chat_session(session_name):
    """Load chat history from a session file."""
    st.session_state.chat_history = load_chat_history(session_name)
    st.session_state.current_session = session_name

def save_chat_history(session_name, history):
    """Save chat history to a file."""
    filename = f"chat_history_{session_name.replace(' ', '_')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for message in history:
            f.write(f"{message['role']}: {message['content']}\\n")

def load_chat_history(session_name):
    """Load chat history from a session file."""
    filename = f"chat_history_{session_name.replace(' ', '_')}.txt"
    history = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                role, content = line.split(":", 1)
                history.append({"role": role.strip(), "content": content.strip()})
        return history
    except FileNotFoundError:
        return []

def clear_chat_history(session_name):
    """Clear chat history file for the given session."""
    filename = f"chat_history_{session_name.replace(' ', '_')}.txt"
    if os.path.exists(filename):
        os.remove(filename)
