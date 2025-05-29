# llm_system/core/database.py
from typing import Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from logger import get_logger
log = get_logger(name="core_database")


class VectorDB:
    """A class to manage the vector database using FAISS and Ollama embeddings.

    Args:
        embed_model (str): The name of the Ollama embeddings model to use.
        persist_path (str, optional): Path to the persisted FAISS database. If None, a new DB is created.
        retriever_num_docs (int): Number of documents to retrieve for similarity search.
        verify_connection (bool): Whether to verify the connection to the embeddings model.

    ## Functions:
        + `get_embeddings()`: Returns the Ollama embeddings model.
        + `get_vector_store()`: Returns the FAISS vector store.
        + `get_retriever()`: Returns the retriever configured for similarity search.
    """

    def __init__(
        self, embed_model: str,
        persist_path: Optional[str] = None,
        retriever_num_docs: int = 5,
        verify_connection: bool = False
    ):
        log.info(
            f"Initializing VectorDB with '{embed_model}' embeddings, path='{persist_path}', k={retriever_num_docs} docs."
        )

        # Here, I have configured the model to be loaded on CPU completely.
        # Reason: Ollama keeps alternately loading and unloading the LLM/Emb model on GPU.
        # Solution: Load the LLM on GPU and the Embedding model on CPU 100%.
        self.embeddings = OllamaEmbeddings(model=embed_model, num_gpu=0)

        if verify_connection:
            try:
                self.embeddings.embed_documents(['a'])
                log.info(f"Embeddings model '{embed_model}' initialized and verified.")

            except Exception as e:
                log.error(f"Failed to initialize Embeddings: {e}")
                raise RuntimeError(f"Couldn't initialize Embeddings model '{embed_model}'") from e
        else:
            log.warning(f"Embeddings '{embed_model}' initialized without connection verification.")

        # If no persisted DB path is provided, create a new FAISS DB
        if persist_path is None:
            dummy_doc = Document(page_content="Hello World!")
            self.db = FAISS.from_documents([dummy_doc], embedding=self.embeddings)
            log.info("Created a new FAISS vector store with a dummy document.")

        else:
            self.db = FAISS.load_local(
                persist_path, self.embeddings, allow_dangerous_deserialization=True)
            log.info(f"Loaded FAISS vector store from path '{persist_path}'.")

        self.retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": retriever_num_docs})

        log.info(f"Created retriever with k={retriever_num_docs}.")

    def get_embeddings(self):
        log.info("Returning the Ollama embeddings model.")
        return self.embeddings

    def get_vector_store(self):
        log.info("Returning the FAISS vector store.")
        return self.db

    def get_retriever(self):
        log.info("Returning the retriever for similarity search.")
        return self.retriever
