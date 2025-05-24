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
DOC_CHUNK_SIZE = 750
DOC_NUM_COUNT = 3000 // DOC_CHUNK_SIZE

TOKENS_PER_SEC = 50

# ------------------------------------------------------------------------------
# Dummy responses of LLM
# ------------------------------------------------------------------------------

# Resp in increasing order of length:
llm_dummy_responses = [
    "Hey there! 👋  \nI'm Gemma-3, a language model developed by the Gemma team at Google. I'm here to help you with questions and create interesting text. I’m still improving and learning each day. Let’s explore and have some fun together! 🤖",

    "Hello 👋!  \nI'm Gemma-3 😎, a large language model created by the Gemma team at Google-Deepmind. I’m here to assist you with a wide range of tasks, from answering your questions to generating creative text formats. My goal is to provide helpful and informative responses. I'm still under development, and I’m learning new things every day! I’m excited to explore with you. Let's see what we can create! 🤖✨",

    "Hello 👋!  \nI'm Gemma-3 😎, a powerful language model developed by the talented folks at Google-Deepmind. I'm here to help you out with a wide range of tasks—whether it’s answering complex questions, crafting detailed explanations, writing stories, poems, or even generating code snippets. I strive to be informative, creative, and engaging in every response I give.  \n\nI'm constantly learning, improving, and adapting to serve you better. Even though I'm still a work in progress, I'm pretty good at what I do! 😄  \n\nFeel free to test my capabilities—ask me anything, challenge me, or just chat. Let’s collaborate, learn new things, and build something awesome together. Ready when you are! 🚀🤖✨"
]
