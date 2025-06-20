# RAG System

A Retrieval-Augmented Generation (RAG) system for processing and querying documents using LangChain, FAISS, and Ollama.

## Project Overview

This project implements a complete RAG pipeline for document processing, embedding, retrieval, and question answering. It's designed to handle PDF documents and provide accurate answers based on the document content.

## Installation & Setup


### 1. Clone the Repository
```bash
git clone https://github.com/jeetKhanpara/RAG_system.git
cd RAG_system
```

### 2. Python Environment Setup
```bash
python3.10 -m venv .venv # Create virtual environment

source .venv/bin/activate # Activate virtual environment

pip install -r requirements.txt # Install dependencies
```

### 3. Install Ollama (for Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh --> 
# this will take some time on first as it will download ollama models

# Start Ollama service
ollama serve

# Pull required model (in a new terminal)
ollama pull llama2:7b # this will also take some time
# Alternative models: mistral, llama2, codellama
```

### Main Configuration (`config/config.py`)
```python
#place your pdf inside data folder and change the name here
FILE_PATH = './data/10050-medicare-and-you_0.pdf'  # Your PDF file path

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Embedding model

EMBEDDING_MODEL_PATH = './faiss_index'     # Vector store path

LLM_MODEL_NAME = 'llama2:7b '  # LLM model name
QUERY = 'What are the important deadlines for Medicare enrollment?'  # Default query, you can change your query here
```

## Usage

### FastAPI Usage

**Step 1: Start Ollama Server**
```bash
ollama serve
```

**Step 2: Start the FastAPI Server**
```bash
python main.py
```

The server will start on `http://localhost:8000`

**Access interactive API docs:** `http://localhost:8000/docs`

**Step 3: Test the API**
```bash
python test_api.py
```

**To change queries:** Edit the `sample_questions` list in `test_api.py`

**Important Notes:**
- You first need to run `main.py` and keep the FastAPI server running
- Keep the Ollama server up to generate queries

**Available Endpoints:**

1. **Health Check**
   - URL: `GET http://localhost:8000/health`
   - Description: Check if retriever and LLM model are initialized

2. **Ask Question**
   - URL: `POST http://localhost:8000/query`
   - Body: `{"question": "Your question here"}`

3. **System Info**
   - URL: `GET http://localhost:8000/info`
   - Description: Get system configuration information

4. **Interactive Documentation**
   - URL: `http://localhost:8000/docs`
   - Description: Swagger UI for testing all endpoints

