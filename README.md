# SweetingTech RAG Chat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LlamaIndex, and LlamaCPP (for connecting to LM Studio). This chatbot allows you to upload documents of various types (PDF, DOCX, TXT, XLSX, images, etc.) and ask questions based on their content. It provides session management, local index persistence, and customizable system instructions.

## Features

*   **Multi-Document Support:**  Ingest and process PDFs, DOCX, TXT, XLSX, images, and HTML files.
*   **Local LLM Integration:**  Connect to a local LM Studio instance for inference.
*   **Retrieval-Augmented Generation:**  Combines document retrieval with LLM generation for accurate and informative answers.
*   **Session Management:**  Create and manage multiple chat sessions.
*   **Local Index Persistence:**  Save and load the LlamaIndex index to avoid re-indexing documents on every run.
*   **Customizable System Instructions:**  Define system instructions (system prompt) to control the AI's behavior.
*   **Short term memory:** Remember previous conversations.

## Project Layout

The project has the following structure:

*   `chatbot_gui.py`: This file contains the Streamlit application code.
*   `chroma_db.py`: This file contains the code for interacting with the Chroma database.
*   `document_loader.py`: This file contains the code for loading documents from different file types.
*   `indexing.py`: This file contains the code for creating and querying the index.
*   `session_manager.py`: This file contains the code for managing chat sessions.
*   `data/`: This directory contains the documents that are uploaded by the user.
*   `vector_index.json`: This file contains the persisted index.
*   `embedding_model/`: This directory contains the embedding models.

## Embedding Folder

The `embedding_model` folder is intended to store the embedding models used by the application. While the folder itself is tracked by Git, the models inside it are not. This is because the models can be large and are not essential for the codebase to function. You can download and place your desired embedding models in this folder.

## Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.7+:** Python must be installed on your system.
*   **LM Studio:** Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/).
*   **Tesseract OCR (Optional):** Required for extracting text from images. See the Installation section below.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SweetingTech/RagChat.git
    cd RagChat
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **On Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```

    *   **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

4.  **Install the required packages:**

    ```bash
    pip install llama-index pypdf pillow python-docx unstructured openpyxl beautifulsoup4 requests streamlit chromadb
    ```

5.  **Install Tesseract OCR (Optional):**

    Tesseract OCR is required for extracting text from images. Follow these steps:

    *   **Windows:**
        1.  Download the Tesseract installer from a reputable source (e.g., [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)).
        2.  Run the installer.
        3.  **Important:** During installation, make sure to add Tesseract to your system's `PATH` environment variable. The installer may offer to do this automatically. If not, you'll need to add the Tesseract installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your `PATH` manually.
    *   **macOS:**

        ```bash
        brew install tesseract
        ```

    *   **Linux (Debian/Ubuntu):**

        ```bash
        sudo apt update
        sudo apt install tesseract-ocr
        ```

    *   **Linux (Fedora/CentOS/RHEL):**

        ```bash
        sudo dnf install tesseract
        ```

    After installing Tesseract, you might need to configure `pytesseract` to find the Tesseract executable. You can do this by setting the `tesseract_cmd` variable in your Python code if it isn't automatically detected:

    ```python
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with your Tesseract path
    ```

## Configuration

1.  **LM Studio Model Path:**

    Update the `MODEL_PATH` variable in the `streamlit_app.py` file to point to the path of your local LLM model in LM Studio (e.g., `"models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"`). Make sure LM Studio is running and serving the model.

2.  **Embedding Model:**

    The default embedding model is `sentence-transformers/all-mpnet-base-v2`. You can change this by modifying the `EMBEDDING_MODEL` variable in `streamlit_app.py`.

3.  **Data Directory:**

    Create a directory named `data` in the project root to store your documents. The chatbot will automatically load and index the files in this directory.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Access the application in your browser:**

    Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Upload documents:**

    Use the file uploader to upload the documents you want to use for querying. The application will automatically create an index of the documents.

4.  **Ask questions:**

    Enter your question in the chat input box and press Enter or click the "Send" button.

5.  **Session management:**

    Use the sidebar to create new chat sessions or load existing ones.

6.  **Customize system instructions:**

    Modify the system instructions in the sidebar to control the AI's behavior.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs or feature requests.

## Contact

SweetingTech - [https://github.com/SweetingTech/](https://github.com/SweetingTech/)
