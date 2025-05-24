# llm_system/core/database.py
from typing import Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class VectorDB:
    """A class to manage the vector database using FAISS and Ollama embeddings.
    
    Args:
        embed_model (str): The name of the Ollama embeddings model to use.
        persist_path (str, optional): Path to the persisted FAISS database. If None, a new DB is created.
        retriever_num_docs (int): Number of documents to retrieve for similarity search.

    ## Functions:
        + `get_embeddings()`: Returns the Ollama embeddings model.
        + `get_vector_store()`: Returns the FAISS vector store.
        + `get_retriever()`: Returns the retriever configured for similarity search.
    """
    def __init__(
        self, embed_model: str,
        persist_path: Optional[str] = None,
        retriever_num_docs: int = 20
    ):
        # Here, I have configured the model to be loaded on CPU completely.
        # Reason: Ollama keeps alternately loading and unloading the LLM/Emb model on GPU.
        # Solution: Load the LLM on GPU and the Embedding model on CPU 100%.
        self.embeddings = OllamaEmbeddings(model=embed_model, num_gpu=0)

        # If no persisted DB path is provided, create a new FAISS DB
        if persist_path is None:
            dummy_doc = Document(page_content="Hello World!")
            self.db = FAISS.from_documents(
                [dummy_doc], embedding=self.embeddings)

        else:
            self.db = FAISS.load_local(
                persist_path, self.embeddings, allow_dangerous_deserialization=True)

        self.retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": retriever_num_docs})

    def get_embeddings(self):
        return self.embeddings

    def get_vector_store(self):
        return self.db

    def get_retriever(self):
        return self.retriever
