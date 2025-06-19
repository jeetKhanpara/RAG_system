from langchain_community.vectorstores import FAISS
import os
path = './faiss_index'
class VectorStore:
    def __init__(self,documents,embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model

    def load_vector_store(self):
        if not os.path.exists(path):
            print('creating new vector store')
            vector_store = FAISS.from_documents(self.documents,self.embedding_model)
            vector_store.save_local('faiss_index')
            return vector_store
        else:
            print('using existing vector store')
            vector_store = FAISS.load_local(path,self.embedding_model,allow_dangerous_deserialization=True)
            return vector_store
        