# from langchain_openai import OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self,model_name:str):
        self.model_name = model_name

    def load_model(self):
        embedding_model = HuggingFaceEmbeddings(
               model_name=self.model_name
        )
        return embedding_model