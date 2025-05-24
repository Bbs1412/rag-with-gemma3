# This file has all the configuration part of AI System

import os
# from dotenv import load_dotenv
# load_dotenv()


# ------------------------------------------------------------------------------
# Main:
# ------------------------------------------------------------------------------

LLM_MODEL_NAME = "gemma3:latest"
EMB_MODEL_NAME = "mxbai-embed-large:latest"

MAX_CONTENT_SIZE = 14000
LLM_TEMPERATURE = 0.75

DOC_CHUNK_SIZE = 750
DOC_NUM_COUNT = 3000 // DOC_CHUNK_SIZE

TOKENS_PER_SEC = 50
BATCH_TOKENS = 2