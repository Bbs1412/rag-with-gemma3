"""Contains a function to split text into smaller chunks."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_system.config import DOC_CHAR_LIMIT, DOC_OVERLAP_NO

from logger import get_logger
log = get_logger(name="utils_splitter")


def split_text(
        documents: List[Document],
        chunk_size: int = DOC_CHAR_LIMIT,
        chunk_overlap: int = DOC_OVERLAP_NO
) -> tuple[bool, List[Document], str]:
    """Splits a list of Document objects into smaller chunks.

    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters that overlap between chunks.

    Returns:
        tuple[bool, List[Document], str]: A tuple containing:
            - bool: True if the documents were split successfully, False otherwise.
            - List[Document]: A list of Document objects containing the split text.
            - str: Message indicating the result of the splitting operation.
    """

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            log.warning("No documents were split. Please check the input documents.")
            return True, [], "No documents were split. Please check the input documents."
        
        log.info(f"Successfully split {len(documents)} documents into {len(split_docs)} chunks.")
        return True, split_docs, "Documents split successfully."

    except Exception as e:
        log.error(f"Error splitting documents: {e}")
        return False, [], f"Error splitting documents: {e}"


if __name__ == "__main__":
    # Example usage
    example_docs = [
        Document(page_content="This is a sample document. " * 10),
        Document(page_content="Another document with some text. " * 5),
        Document(page_content="Yet another document with different content. " * 3)
    ]

    status, split_documents, message = split_text(example_docs, chunk_size=100, chunk_overlap=10)

    for i, doc in enumerate(split_documents):
        print(f"Chunk {i+1}: {doc.page_content}")
        # Print first 50 characters of each chunk
        # print(f"Chunk {i+1}: {doc.page_content[:50]}...")
