"""
config.py - Central configuration for AI System.

This module stores all configurable constants related to:
- LLMs (chat, summarization)
- Embeddings
- Chunking and content limits
- Verification checks
- Dummy response simulator
"""

# import os
# from dotenv import load_dotenv
# load_dotenv()


# Model configuration::
LLM_CHAT_MODEL_NAME: str = "gemma3:latest"              # Chatting model
LLM_CHAT_TEMPERATURE: float = 0.75
LLM_SUMMARY_MODEL_NAME: str = "gemma3:latest"           # History Summarization model
LLM_SUMMARY_TEMPERATURE: float = 0.5
EMB_MODEL_NAME: str = "mxbai-embed-large:latest"        # Embeddings model

# The max token count which shall be allowed after 'chat_history + input + context'.
MAX_CONTENT_SIZE: int = 14000


# Verification configuration:
#   - Whether to immediately verify the connection to
#   - the LLM models and the Embeddings models after initialization.
#   - Useful specifically for Ollama models, as they can be loaded on GPU or CPU.
VERIFY_LLM_CONNECTION: bool = False
VERIFY_EMB_CONNECTION: bool = False


# Document Chunking properties:
DOC_CHAR_LIMIT: int = 2000                              # Char limit for each doc.
DOC_OVERLAP_NO: int = 250                               # Char limit for chunk overlap.


# Document Retrieval properties:
DOC_TOKEN_SIZE: int = DOC_CHAR_LIMIT // 4               # Appx number of tokens in each doc.
DOCS_NUM_COUNT: int = 3000 // DOC_TOKEN_SIZE            # Max num of docs to retrieve.


# Database:
VECTOR_DB_PERSIST_DIR: str = "user_faiss"            # Path to persist the vector DB.
VECTOR_DB_INDEX_NAME: str = "index.faiss"               # Name of the vector DB file.

# Dummy response mode properties:
TOKENS_PER_SEC: int = 50                                # num of tokens yielded per sec
BATCH_TOKEN_PS: int = 2                                 # num of tokens yielded in each batch
