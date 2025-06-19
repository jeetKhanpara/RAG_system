# RAG System

A Retrieval-Augmented Generation (RAG) system for processing and querying documents using LangChain, FAISS, and Ollama.

## Project Overview

This project implements a complete RAG pipeline for document processing, embedding, retrieval, and question answering. It's designed to handle PDF documents and provide accurate answers based on the document content.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project_mi
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Ollama (for Local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh --> this will take some time on first as it will download ollama models

# Start Ollama service
ollama serve

# Pull required model (in a new terminal)
ollama pull phi3 
# Alternative models: mistral, llama2, codellama
```

### 4. Environment Configuration

# Other configurations
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=phi3
```

### 5. Data Preparation
```bash
# Place your PDF documents in the data/ directory
cp your_document.pdf data/

# Update configuration in config/config.py
# Modify FILE_PATH to point to your document
```

## Configuration

### Main Configuration (`config/config.py`)
```python
FILE_PATH = './data/10050-medicare-and-you_0.pdf'  # Your PDF file path
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'          # Embedding model
EMBEDDING_MODEL_PATH = './faiss_index'              # Vector store path
LLM_MODEL_NAME = 'phi3'                            # LLM model name
QUERY = 'What are the important deadlines for Medicare enrollment?'  # Default query
```

### Available Models

**Embedding Models:**
- `all-MiniLM-L6-v2` (~90MB) - Current default
- `intfloat/e5-large-v2` (~440MB) - Higher quality
- `BAAI/bge-m3` (~1.5GB) - Multilingual support

**LLM Models (via Ollama):**
- `phi3` (~4.1GB) - Current default
- `mistral` (~4.1GB) - Good performance
- `llama2` (~3.8GB) - Meta's model
- `codellama` (~3.8GB) - Code-focused

## Usage

### Basic Usage
```bash
# Run the main retrieval script
python retrieval.py
```

### Web API (FastAPI)
```bash
# Uncomment FastAPI code in retrieval.py
# Then run:
uvicorn retrieval:app --host 0.0.0.0 --port 8000
```

### Jupyter Notebook
```bash
# For interactive development
jupyter notebook practice.ipynb
```

## Component Details

### 1. Document Loading (`components/loader.py`)
- Uses PyMuPDF for PDF processing
- Extracts text and metadata from PDF pages
- Returns LangChain Document objects

### 2. Text Splitting (`components/splitter.py`)
- RecursiveCharacterTextSplitter with 7000 chunk size
- 20 character overlap for context continuity
- Preserves metadata across chunks

### 3. Embedding (`components/embedder.py`)
- HuggingFace embeddings integration
- Supports various embedding models
- Generates vector representations

### 4. Vector Store (`components/vectore_store.py`)
- FAISS for similarity search
- Automatic index creation and loading
- Persistent storage in `faiss_index/` directory

### 5. Retrieval & Generation
- Similarity search with configurable k-value
- Prompt template for context-aware responses
- LLM integration via Ollama

## Performance Optimization

### Current Optimizations
- Vector store caching/reuse
- Configurable chunk sizes
- Efficient embedding model selection

### Recommended Improvements
```python
# Additional dependencies for optimization
pydantic==2.0+          # Data validation
loguru==0.7+            # Better logging
redis==4.5+             # Caching (optional)
```

### Scaling Considerations
- **Small Scale**: Current setup (suitable for <1000 documents)
- **Medium Scale**: Add Redis caching, optimize chunking
- **Large Scale**: Consider Pinecone/Weaviate, load balancing

## Development

### Code Quality
```bash
# Install development dependencies
pip install black flake8 pytest

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

### Project Structure Best Practices
- Modular component design
- Configuration separation
- Clear dependency management
- Comprehensive documentation

## Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Ensure Ollama is running
ollama serve

# Check model availability
ollama list
```

**2. Memory Issues**
- Reduce chunk size in `splitter.py`
- Use smaller embedding model
- Increase system RAM

**3. Model Download Issues**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Use VPN if needed for model downloads
```

**4. FAISS Index Issues**
```bash
# Remove existing index
rm -rf faiss_index/

# Regenerate index
python retrieval.py
```

## Production Deployment

### Docker Setup
```dockerfile
# Example Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "retrieval:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# Production environment
export PYTHONPATH=/app
export OLLAMA_HOST=http://ollama:11434
export LOG_LEVEL=INFO
```

### Monitoring
- Health checks for API endpoints
- Metrics collection (Prometheus/Grafana)
- Log aggregation
- Error tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration options
- Ensure all dependencies are properly installed
- Verify system requirements are met
