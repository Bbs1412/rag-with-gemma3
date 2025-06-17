""" A script which will deal with ingestion of new documents into the vector database.
- Currently has file ingestion which supports txt, pdf, and md files.
- Plan to add more file types in the future.
- Plan to add web based ingestion in the future.
"""

from typing import List

from llm_system.utils.loader import load_file
from llm_system.utils.splitter import split_text

# For type hinting
from llm_system.core.database import VectorDB
from langchain_core.embeddings import Embeddings


from logger import get_logger
log = get_logger(name="core_ingestion")


def ingest_file(user_id: str, file_path: str, vectorstore: VectorDB,
                embeddings: Embeddings) -> tuple[bool, List[str], str]:
    """Ingest a file into the vector database. Returns the ids of vector embeddings stored in database.

    Args:
        file_path (str): The absolute path to the file to be ingested.
        db (VectorDB): The vector database instance.
        embeddings (Embeddings): The embeddings model to use for the documents.

    Returns:
        tuple[bool, List[str], str]: A tuple containing:
            - bool: True if ingestion was successful, False otherwise.
            - List[str]: List of document IDs that were ingested.
            - str: Message indicating the result of the ingestion.
    """

    # Load the file and get its content as Document objects:
    status, documents, message = load_file(user_id, file_path)
    # print(status, documents, message)

    if not status:
        return False, [], message

    # Split the documents into smaller chunks:
    status, split_docs, message = split_text(documents)
    if status and not split_docs:
        log.warning(f"No content found in the file: {file_path}")
        return True, [], f"No content found in the file: {file_path}"

    if not status:
        return False, [], message

    # Add the split documents to the vector database:
    try:
        doc_ids = vectorstore.db.add_documents(split_docs, embeddings=embeddings)
        if vectorstore.save_db_to_disk():
            log.info(f"Ingested {len(split_docs)} documents from {file_path} into the vector database.")
            return True, doc_ids, f"Ingested {len(split_docs)} documents successfully."
        else:
            log.error("Failed to save the vector database to disk after ingestion.")
            return False, [], "Failed to save the vector database to disk after ingestion."
            
    except Exception as e:
        log.error(f"Failed to ingest documents: {e}")
        return False, [], f"Failed to ingest documents: {e}"


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain.callbacks.tracers.langchain import wait_for_all_tracers
    load_dotenv()

    # Example usage
    user = "test_user"
    # example_file_path = "../../../GenAI/Data/attention_is_all_you_need_1706.03762v7.pdf"
    example_file_path = "../../../GenAI/Data/speech.md"

    vector_db = VectorDB(embed_model="mxbai-embed-large:latest", persist_path=None)

    status, doc_ids, message = ingest_file(user, example_file_path, vector_db, vector_db.embeddings)
    if status:
        print(doc_ids)
    else:
        print(f"Error: {message}")

    # Retrieve the documents to verify ingestion:
    print(
        vector_db.retriever.invoke(
            input="What is the attention mechanism in transformers?",
            filter={"user_id": user}
        )
    )

    print(
        vector_db.retriever.invoke(
            input="What is the attention mechanism in transformers?",
            filter={"user_id": "random"}
        )
    )

    wait_for_all_tracers()
