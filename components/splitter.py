from langchain.text_splitter import RecursiveCharacterTextSplitter

def query_to_chunk_config(query: str):
    if len(query) < 30 or any(q in query.lower() for q in ["when", "what", "how much", "who"]):
        return {"chunk_size": 300, "chunk_overlap": 20}  # precise Q
    elif any(q in query.lower() for q in ["summarize", "overview", "explain", "analyze"]):
        return {"chunk_size": 1000, "chunk_overlap": 100}  # broader context
    else:
        return {"chunk_size": 512, "chunk_overlap": 50}  # default

class Splitter:

    def __init__(self, document: list, query: str = None):
        self.document = document
        self.query = query

    def split_document(self, query: str = None):

        target_query = query if query is not None else self.query
        
        if target_query:
            chunk_config = query_to_chunk_config(target_query)
            chunk_size = chunk_config["chunk_size"]
            chunk_overlap = chunk_config["chunk_overlap"]
            print(f"Using dynamic chunking: size={chunk_size}, overlap={chunk_overlap} for query: '{target_query[:50]}...'")
        else:
            # default
            chunk_size = 800
            chunk_overlap = 80
            print(f"Using default chunking: size={chunk_size}, overlap={chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = splitter.split_documents(self.document)

        return documents