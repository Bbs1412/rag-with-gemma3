""" Database Module for LLM System
- Contains the `VectorDB` class to manage a vector database using FAISS and Ollama embeddings.
- Provides methods to initialize the database, retrieve embeddings, and perform similarity searches.
"""

import os
from typing import Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import ConfigurableField

# For type hinting
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

# config:
from llm_system.config import VECTOR_DB_PERSIST_DIR, VECTOR_DB_INDEX_NAME

from logger import get_logger
log = get_logger(name="core_database")


class VectorDB:
    """A class to manage the vector database using FAISS and Ollama embeddings.

    Args:
        embed_model (str): The name of the Ollama embeddings model to use.
        retriever_num_docs (int): Number of documents to retrieve for similarity search.
        verify_connection (bool): Whether to verify the connection to the embeddings model.
        persist_path (str, optional): Path to the persisted FAISS database. If None, a new DB is created.
        index_name (str, optional): Name of the FAISS index file. Defaults to "index.faiss".

    ## Functions:
        + `get_embeddings()`: Returns the Ollama embeddings model.
        + `get_vector_store()`: Returns the FAISS vector store.
        + `get_retriever()`: Returns the retriever configured for similarity search.
    """

    def __init__(
        self, embed_model: str,
        retriever_num_docs: int = 5,
        verify_connection: bool = False,
        persist_path: Optional[str] = VECTOR_DB_PERSIST_DIR,
        index_name: Optional[str] = VECTOR_DB_INDEX_NAME
    ):
        self.persist_path: Optional[str] = persist_path
        self.index_name: Optional[str] = index_name

        log.info(
            f"Initializing VectorDB with embeddings='{embed_model}', path='{persist_path}', k={retriever_num_docs} docs."
        )

        # Here, I have configured the model to be loaded on CPU completely.
        # Reason: Ollama keeps alternately loading and unloading the LLM/Emb model on GPU.
        # Solution: Load the LLM on GPU and the Embedding model on CPU 100%.
        self.embeddings = OllamaEmbeddings(
            base_url="http://host.docker.internal:11434",  # Use host's IP for Docker
            model=embed_model, num_gpu=0, keep_alive=-1
        )

        if verify_connection:
            try:
                self.embeddings.embed_documents(['a'])
                log.info(f"Embeddings model '{embed_model}' initialized and verified.")

            except Exception as e:
                log.error(f"Failed to initialize Embeddings: {e}")
                raise RuntimeError(f"Couldn't initialize Embeddings model '{embed_model}'") from e
        else:
            log.warning(f"Embeddings '{embed_model}' initialized without connection verification.")

        # Create a dummy document to initialize the FAISS vector store:
        dummy_doc = Document(
            page_content="Hello World!",
            metadata={"user_id": "public", 'source': "test document"}
        )

        # Load faiss from disk:
        if persist_path and index_name:
            database_file = os.path.join(persist_path, index_name)

            if not os.path.exists(database_file):
                self.db = FAISS.from_documents([dummy_doc], embedding=self.embeddings)
                self.db.save_local(persist_path)
                log.info("Created a new FAISS vector store on disk with a dummy document.")
            else:
                log.info(f"Found existing FAISS vector store at '{database_file}'.")
                self.db = FAISS.load_local(
                    persist_path, self.embeddings, allow_dangerous_deserialization=True)

        # Create one temp, in memory, FAISS vector store:
        else:
            self.db = FAISS.from_documents([dummy_doc], embedding=self.embeddings)
            log.info("Created a new FAISS vector store in memory with a dummy document.")

        # self.retriever = self.db.as_retriever(
        #     search_type="similarity",
        #     search_kwargs={"k": retriever_num_docs, "filter":{"user_id": "public"}},
        # )
        # log.info(f"Created retriever with k={retriever_num_docs}.")

        # Simple retriever does not have way to pass some filters with rag_chain.invoke()
        # Basically no way to pass args at runtime
        # Hence, using configurable retriever:
        # https://github.com/langchain-ai/langchain/issues/9195#issuecomment-2095196865
        retriever = self.db.as_retriever()
        configurable_retriever = retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )

        # call it like this:
        # configurable_retriever.invoke(
        #     input="What is the Sun?",
        #     config={"configurable": {
        #         "search_kwargs": {
        #             "k": 5,
        #             "search_type": "similarity",
        #             # And here comes the main thing:
        #             "filter": {
        #                 "$or": [
        #                     {"user_id": "curious_cat"},
        #                     {"user_id": "public"}
        #                 ]
        #             },
        #         }
        #     }}
        # )

        self.retriever = configurable_retriever
        log.info(f"Created configurable retriever.")

    def get_embeddings(self) -> Embeddings:
        log.info("Returning the Embeddings model instance.")
        return self.embeddings

    def get_vector_store(self) -> VectorStore:
        log.info("Returning the FAISS vector store instance.")
        return self.db

    def get_retriever(self) -> VectorStoreRetriever:
        log.info("Returning the retriever for similarity search.")
        return self.retriever  # type: ignore[return-value]

    def save_db_to_disk(self) -> bool:
        """Saves the current vector store to disk if a persist path is set.
        Returns:
            bool: True if the vector store was saved successfully, False otherwise.
        """

        if self.persist_path and self.index_name:
            try:
                # Somehow, loading needs 'index.faiss', but saving needs only 'index'.
                # index_base_name = self.index_name[:-6] if self.index_name.endswith('.faiss') else self.index_name
                if self.index_name.endswith('.faiss'):
                    index_base_name = self.index_name[:-6]
                else:
                    index_base_name = self.index_name

                self.db.save_local(self.persist_path, index_name=index_base_name)
                log.info(f"Vector store saved to disk at '{self.persist_path}/{self.index_name}'.")
                return True
            except Exception as e:
                log.error(f"Failed to save vector store to disk: {e}")
                return False
        else:
            log.warning("Skipped saving to disk as no persist path is set.")
            return True
