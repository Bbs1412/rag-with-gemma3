"""Module dealing specifically with loading files into Document objects.
Contains the `load_file` function to load text, PDF, and markdown files.
Planning to add more file types in the future.

## For testing:
- Run this file from `server` folder as:
- `python -m llm_system.utils.loader`
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from logger import get_logger
log = get_logger(name="doc_loader")


def load_file(user_id: str, file_path: str) -> tuple[bool, List[Document], str]:
    """Load a file and return its content as a list of Document objects. Usually one document per page.

    Args:
        user_id (str): The ID of the user who is loading the file.
        file_path (str): The absolute path to the file to be loaded.

    Returns:
        tuple[bool, List[Document], str]: A tuple containing:
            - bool: True if the file was loaded successfully, False otherwise.
            - List[Document]: A list of Document objects containing the file's content.
            - str: Message indicating the result of the loading operation.
    """

    # Planning to add many types in future, but for now, only txt and pdf are supported:
    file_extension = file_path.split('.')[-1].lower()

    if file_extension not in ['txt', 'pdf', "md"]:
        log.error(f"Unsupported file type: {file_extension}.")
        return False, [], f"Unsupported file type: {file_extension}. Supported types are: txt, pdf."

    if file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')

    elif file_path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)

    else:
        loader = PyMuPDFLoader(file_path, extract_images=False)

    # Load the file and return the documents
    file_content = loader.load()

    # Add user metadata to each doc:
    for doc in file_content:
        doc.metadata['user_id'] = user_id
        # Since i am exposing the retrieved docs to UI
        # Hide full server file path if its there:
        if 'file_path' in doc.metadata:
            # doc.metadata['file_path'] = doc.metadata['file_path'].split('/')[-1]
            doc.metadata['file_path'] = os.path.basename(doc.metadata['file_path'])

        if 'source' in doc.metadata:
            # If it is not local file, keep source as is:
            if "www." in doc.metadata['source'] or "http" in doc.metadata['source']:
                continue
            # If it is local file, keep only the file name:
            else:
                # if '/' in doc.metadata['source']:
                #     doc.metadata['source'] = doc.metadata['source'].split('/')[-1]
                doc.metadata['source'] = os.path.basename(doc.metadata['source'])

    if not file_content:
        log.error(f"No content found in the file: {file_path}")
        return True, [], f"No content found in the file: {file_path}"
        # raise ValueError(f"No content found in the file: {file_path}")

    log.info(f"Loaded {len(file_content)} documents from {file_path} for user {user_id}.")
    return True, file_content, f"Loaded {len(file_content)} documents."


if __name__ == "__main__":
    # Example usage
    try:
        status, docs, message = load_file(
            user_id="test_user",
            file_path="../../../GenAI/Data/attention_is_all_you_need_1706.03762v7.pdf"
            # file_path="../../../GenAI/Data/speech.txt"
            # file_path="../../../GenAI/Data/speech.md"
        )

        print(status)
        print(message)
        print(len(docs))

        for ind, doc in enumerate(docs[:3]):
            print("\n")
            print(repr(doc))

    except Exception as e:
        print(f"Error loading file: {e}")
