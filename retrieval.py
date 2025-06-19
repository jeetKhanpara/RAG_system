from config.config import QUERY,FILE_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH
from dotenv import load_dotenv
from components.loader import Loader
from components.splitter import Splitter
from components.embedder import Embedder
from components.vectore_store import VectorStore
import time
import os

load_dotenv()

def build_vector_store(query: str = None):
   
    start_time = time.time()
    
    print("=== Starting Vector Store Creation ===")
    
    # Loading the document
    print("Loading document...")
    loader = Loader(FILE_PATH)
    document = loader.load_document()  # list of langchain documents (page_content and metadata)
    print("===Document loaded===")

    # Splitting the document with dynamic chunking
    print("Splitting document...")
    splitter = Splitter(document, query)
    documents = splitter.split_document(query)  # list of langchain documents (page_content and metadata) but this time larger in size
    print(f"===Document split into {len(documents)} chunks===")

    # Initialize embedding model
    print("Loading embedding model...")
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    embedding_model = embedder.load_model()
    print("===Embedding model loaded===")

    # Create and save vector store
    print("Creating vector store...")
    vector_store = VectorStore(documents, embedding_model)
    loaded_vector_store = vector_store.load_vector_store()
    print("===Vector store created and saved===")

    end_time = time.time()
    print(f"=== Vector Store Creation Complete ===")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return loaded_vector_store

def rebuild_vector_store_for_query(query: str):
    """
    Rebuild the vector store with chunking optimized for a specific query.
    Use this when you want to optimize the vector store for a particular type of query.
    
    Args:
        query (str): The query to optimize chunking for
    """
    print(f"=== Rebuilding Vector Store for Query: '{query[:50]}...' ===")
    
    # Remove existing vector store to force rebuild
    if os.path.exists(EMBEDDING_MODEL_PATH):
        import shutil
        shutil.rmtree(EMBEDDING_MODEL_PATH)
        print("Removed existing vector store")
    
    # Build new vector store with query-specific chunking
    return build_vector_store(query)

def get_retriever():
    """
    Load existing vector store and return a retriever.
    This function can be called multiple times for different queries.
    """
    print("Loading existing vector store...")
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    embedding_model = embedder.load_model()
    
    vector_store = VectorStore([], embedding_model)  # Empty documents since we're loading existing
    loaded_vector_store = vector_store.load_vector_store()
    
    # Make vector store a retriever
    retriever = loaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("✓ Retriever initialized")
    
    return retriever

if __name__ == "__main__":
    build_vector_store()
    