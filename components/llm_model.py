# from langchain_openai import ChatOpenAI

# class Model:
#     def __init__(self,llm_model_name):
#         self.llm_model_name = llm_model_name

#     def load_llm_model(self):
#         llm = ChatOpenAI(model=self.llm_model_name)
#         return llm 

from langchain.llms import Ollama

class Model:
    def __init__(self,llm_model_name):
        self.llm_model_name = llm_model_name

    def load_llm_model(self):
        llm = Ollama(model=self.llm_model_name)
        return llm 

