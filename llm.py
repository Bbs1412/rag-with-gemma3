from time import sleep
from log import Logger
from langchain_ollama import ChatOllama
from typing import Literal, Generator, Union

llm = ChatOllama(model="gemma3:1b")

logger = Logger(name="LLM", log_to_console=True)
logger.log("LLM initialized.", level='info')

# Lesson learned: Never use return and yield in the same function.
# Python will always return generators !

dummy_resp = "Hello 👋!  \nI'm Gemma-3 😎, a large language model created by the Gemma team at Google-Deepmind. I’m here to assist you with a wide range of tasks, from answering your questions to generating creative text formats. My goal is to provide helpful and informative responses. I'm still under development, and I’m learning new things every day! I’m excited to explore with you. Let's see what we can create! 🤖✨"


def get_response(prompt: str = "Hello!", dummy: bool = False) -> str:
    """Returns a complete response as a string (non-streaming mode)."""
    if dummy:
        logger.log(f"Type: Dummy, Stream: False", level='info')
        return dummy_resp

    else:
        logger.log(f"Type: LLM, Stream: False", level='info')
        return str(llm.invoke(prompt).content)


def get_response_stream(prompt: str = "Hello!", dummy: bool = False) -> Generator[str, None, None]:
    """Yields the response in chunks (streaming mode)."""

    if dummy:
        logger.log(f"Type: Dummy, Stream: True", level='info')
        for i in range(0, len(dummy_resp), 3):
            yield dummy_resp[i:i + 3]
            sleep(0.05)
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
