import os
from time import sleep
from log import Logger
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Literal, Generator, Union
from random import choice

load_dotenv()
llm = ChatOllama(model="gemma3:1b")
logger = Logger(name="LLM", log_to_console=True)

# Lesson learned: Never use return and yield in the same function.
# Python will always return generators !

dummy_resp = [
    "Hey there! 👋  \nI'm Gemma-3, a language model developed by the Gemma team at Google. I'm here to help you with questions and create interesting text. I’m still improving and learning each day. Let’s explore and have some fun together! 🤖",
    "Hello 👋!  \nI'm Gemma-3 😎, a large language model created by the Gemma team at Google-Deepmind. I’m here to assist you with a wide range of tasks, from answering your questions to generating creative text formats. My goal is to provide helpful and informative responses. I'm still under development, and I’m learning new things every day! I’m excited to explore with you. Let's see what we can create! 🤖✨",
    "Hello 👋!  \nI'm Gemma-3 😎, a powerful language model developed by the talented folks at Google-Deepmind. I'm here to help you out with a wide range of tasks—whether it’s answering complex questions, crafting detailed explanations, writing stories, poems, or even generating code snippets. I strive to be informative, creative, and engaging in every response I give.  \n\nI'm constantly learning, improving, and adapting to serve you better. Even though I'm still a work in progress, I'm pretty good at what I do! 😄  \n\nFeel free to test my capabilities—ask me anything, challenge me, or just chat. Let’s collaborate, learn new things, and build something awesome together. Ready when you are! 🚀🤖✨"
]


def get_response(prompt: str = "Hello!", dummy: bool = False) -> str:
    """Returns a complete response as a string (non-streaming mode)."""
    if dummy:
        logger.log(f"Type: Dummy, Stream: False", level='info')
        return choice(dummy_resp)

    else:
        logger.log(f"Type: LLM, Stream: False", level='info')
        return str(llm.invoke(prompt).content)


def get_response_stream(prompt: str = "Hello!", dummy: bool = False) -> Generator[str, None, None]:
    """Yields the response in chunks (streaming mode)."""

    if dummy:
        batch_tokens = 2
        tokens_per_sec = int(os.environ.get("tokens_per_sec", 45))
        delay = batch_tokens / tokens_per_sec

        logger.log(f"Type: Dummy, Stream: True", level='info')
        dummy_response = choice(dummy_resp)
        for i in range(0, len(dummy_response.split(" ")), batch_tokens):
            yield " ".join(dummy_response.split(" ")[i:i + batch_tokens])+" "
            sleep(delay)

    else:
        logger.log(f"Type: LLM, Stream: True", level='info')
        for chunk in llm.stream(prompt):
            yield chunk.content


if __name__ == "__main__":
    def sep():
        print(f"\n {'-' * 100} \n\n")

    user_message = "What is Gemma-3?"
    logger.log("Seeking response...", level='info')

    sep()
    # Test the dummy response without streaming:
    print(get_response(user_message, dummy=True))

    sep()
    # Test the LLM response without streaming:
    print(get_response(user_message, dummy=False))

    sep()
    # Test the dummy response with streaming:
    response = get_response_stream(user_message, dummy=True)
    for chunk in response:
        print(chunk, end='', flush=True)

    sep()
    # Test the LLM response with streaming:
    response = get_response_stream(user_message, dummy=False)
    for chunk in response:
        print(chunk, end='', flush=True)
    sep()

    logger.log("All tests successful", level='info')
