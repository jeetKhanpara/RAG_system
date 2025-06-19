from langchain_community.document_loaders import PyMuPDFLoader

class Loader:

    def __init__(self,path:str):
        self.path = path

    def load_document(self):
        loader = PyMuPDFLoader(self.path)
        documet = loader.load()

        return documet