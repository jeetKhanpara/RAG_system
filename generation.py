from config.config import LLM_MODEL_NAME, QUERY
from dotenv import load_dotenv
from components.augmentation import Augmentate
from components.llm_model import Model
from retrieval import get_retriever
import time

load_dotenv()

def generate_response(query):

    start_time = time.time()
    
    print(f"=== Generating Response for Query ===")
    print(f"Query: {query}")
    
    # Get retriever from existing vector store
    retriever = get_retriever()
    
    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    retrieved_docs = retriever.invoke(query)
    print(f"===Retrieved {len(retrieved_docs)} relevant documents===")

    # Augmentation part (prompt + llm)
    print("Creating prompt...")
    augmentate = Augmentate(retrieved_docs, query)
    prompt = augmentate.create_prompt()
    print("===Prompt created===")

    # Initialize and use LLM
    print("Initializing LLM model...")
    llm = Model(LLM_MODEL_NAME)
    llm_model = llm.load_llm_model()
    print("===LLM model initialized===")

    print("Generating response...")
    result = llm_model.invoke(prompt)
    print("===Response generated===")

    end_time = time.time()
    
    print(f"\n=== Response ===")
    print(f"Query: {query}")
    print(f"Answer: {result}")
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    
    return result

if __name__ == "__main__":
    # Example usage
    query = QUERY
    generate_response(query)
    
    # Uncomment the line below to run in interactive mode
    # interactive_mode() 