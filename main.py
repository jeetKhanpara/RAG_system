from config.config import FILE_PATH, EMBEDDING_MODEL_NAME, QUERY, LLM_MODEL_NAME
from dotenv import load_dotenv
from components.loader import Loader
from components.splitter import Splitter
from components.embedder import Embedder
from components.vectore_store import VectorStore
from components.augmentation import Augmentate
from components.llm_model import Model
import time
import os
from fastapi import FastAPI

# app = FastAPI()

load_dotenv()

# app.get('/')
# def hello():
#     print("welcome to rag world")

def main():
    start_time = time.time()
    #loading the document
    loader = Loader(FILE_PATH)
    document = loader.load_document() # list of langchain documents (page_content and metadata)
    print("document loaded")

    #splitting the document
    splitter = Splitter(document)
    documents = splitter.split_document() # # list of langchain documents (page_content and metadata) but this time larger in size
    print(f"document splitted into {len(documents)} chunks")

    #initialize embedding model as well as vector store
    print('embedding model is being loaded...')
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    embedding_model = embedder.load_model()
    print("embedding model loaded")

    print('vector sotre is being loaded...')
    vector_store = VectorStore(documents,embedding_model)
    loaded_vector_store = vector_store.load_vector_store()
    print("vector store loaded")

    # make vector store a retrievar
    retriever = loaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("vector store as retrievar is initialized")
    
    retrieved_docs = retriever.invoke(QUERY)

    #Augmentation part (prompt + llm)
    augmentate = Augmentate(retrieved_docs,QUERY)
    prompt = augmentate.create_prompt()
    print("prompt has been created")

    llm = Model(LLM_MODEL_NAME)
    llm_model = llm.load_llm_model()
    print('llm model has been initialized')

    result = llm_model.invoke(prompt)

    print(f'query:{QUERY}\n answer:{result}')

    end_time = time.time()

    print(f"total time taken is {end_time-start_time}")

if __name__ == "__main__":
    main()