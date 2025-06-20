from config.config import FILE_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from components.loader import Loader
from components.splitter import Splitter
from components.embedder import Embedder
from components.vectore_store import VectorStore
from components.augmentation import Augmentate
from components.llm_model import Model
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="RAG API"
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    processing_time: float

# Global retriever object
retriever = None
llm_model = None

def initialize_rag_system():
    """Initialize the RAG system components"""
    global retriever, llm_model
    
    print("Initializing RAG system...")
    start_time = time.time()
    
    # Loading the document
    loader = Loader(FILE_PATH)
    document = loader.load_document()
    print("Document loaded")

    # Splitting the document
    splitter = Splitter(document)
    documents = splitter.split_document()
    print(f"Document split into {len(documents)} chunks")

    # Initialize embedding model as well as vector store
    print('Embedding model is being loaded...')
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    embedding_model = embedder.load_model()
    print("Embedding model loaded")

    print('Vector store is being loaded...')
    vector_store = VectorStore(documents, embedding_model)
    loaded_vector_store = vector_store.load_vector_store()
    print("Vector store loaded")

    # Make vector store a retriever
    retriever = loaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("Vector store as retriever is initialized")
    
    # Initialize LLM model
    llm = Model(LLM_MODEL_NAME)
    llm_model = llm.load_llm_model()
    print('LLM model has been initialized')

    end_time = time.time()
    print(f"Total time taken to initialize RAG system: {end_time-start_time:.2f} seconds")

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system when the FastAPI app starts"""
    initialize_rag_system()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to RAG",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if retriever is None or llm_model is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "status": "healthy",
        "retriever_initialized": retriever is not None,
        "llm_model_initialized": llm_model is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system with a question"""
    if retriever is None or llm_model is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    
    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(request.question)
        
        # Augmentation part (prompt + llm)
        augmentate = Augmentate(retrieved_docs, request.question)
        prompt = augmentate.create_prompt()
        
        # Generate answer
        result = llm_model.invoke(prompt)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return QueryResponse(
            question=request.question,
            answer=str(result),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/info")
async def get_system_info():
    """Get information about the RAG system"""
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "document_path": FILE_PATH,
        "system_initialized": retriever is not None and llm_model is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)