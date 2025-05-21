import os
from log import Logger
from time import sleep
from random import choice
from dotenv import load_dotenv
from typing import Literal, Generator, Union


# Chat:
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# History
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
# Load
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
# Store
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# Retrieve
from langchain.chains import create_retrieval_chain, create_history_aware_retriever


# ------------------------------------------------------------------------------
# Basic Declarations:
# ------------------------------------------------------------------------------

load_dotenv()
logger = Logger(name="LLM", log_to_console=True)

dummy_resp = [
    "Hey there! 👋  \nI'm Gemma-3, a language model developed by the Gemma team at Google. I'm here to help you with questions and create interesting text. I’m still improving and learning each day. Let’s explore and have some fun together! 🤖",
    "Hello 👋!  \nI'm Gemma-3 😎, a large language model created by the Gemma team at Google-Deepmind. I’m here to assist you with a wide range of tasks, from answering your questions to generating creative text formats. My goal is to provide helpful and informative responses. I'm still under development, and I’m learning new things every day! I’m excited to explore with you. Let's see what we can create! 🤖✨",
    "Hello 👋!  \nI'm Gemma-3 😎, a powerful language model developed by the talented folks at Google-Deepmind. I'm here to help you out with a wide range of tasks—whether it’s answering complex questions, crafting detailed explanations, writing stories, poems, or even generating code snippets. I strive to be informative, creative, and engaging in every response I give.  \n\nI'm constantly learning, improving, and adapting to serve you better. Even though I'm still a work in progress, I'm pretty good at what I do! 😄  \n\nFeel free to test my capabilities—ask me anything, challenge me, or just chat. Let’s collaborate, learn new things, and build something awesome together. Ready when you are! 🚀🤖✨"
]


# ------------------------------------------------------------------------------
# Basic Function for development:
# ------------------------------------------------------------------------------

# Lesson learned: Never use return and yield in the same function.
# Python will always return generators !

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


# ------------------------------------------------------------------------------
# Main:
# ------------------------------------------------------------------------------

model_name = "gemma3:latest"
emb_model_name = "mxbai-embed-large:latest"

llm = ChatOllama(model=model_name, num_ctx=14000, temperature=1)
llm_retrieval = ChatOllama(model=model_name, num_ctx=14000, temperature=0.75)

embedding_model = OllamaEmbeddings(model=emb_model_name)

max_context_size = 14000
doc_chunk_size = 750
doc_count = 3000 // doc_chunk_size


# ------------------------------------------------------------------------------
# Chat History:
# ------------------------------------------------------------------------------

chat_histories = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
        logger.log(
            f"Created chat history for session id: `{session_id}`", level='info')
    return chat_histories[session_id]


logger.log(f"Chat histories store initialized", level='info')

# For summary 15k chat + 1k system and all
trim_summary = trim_messages(
    max_tokens=max_context_size - 1000,
    strategy="last", token_counter=llm_retrieval, start_on="human",
    allow_partial=True,  # include_system=True,
)

# For chat 10k chat + 5*1k docs + 1k system and all
trim_chat = trim_messages(
    max_tokens=(max_context_size - doc_chunk_size * doc_count - 1000),
    strategy="last", token_counter=llm, start_on="human",
    allow_partial=True
)


# ------------------------------------------------------------------------------
# Data and Embeddings:
# ------------------------------------------------------------------------------

database = FAISS.from_documents(
    [Document("Hello World!")], embedding=embedding_model)

retriever = database.as_retriever(
    search_type="similarity", search_kwargs={'k': 20})

logger.log(
    f"Database initialized with {emb_model_name} embeddings.", level='info')


# ------------------------------------------------------------------------------
# Prompt Templates:
# ------------------------------------------------------------------------------


# Chat Template:
template_chat = ChatPromptTemplate.from_messages(
    messages=[
        ("system",  (
            "You are a highly knowledgeable and helpful AI assistant.\n"
            "You are provided with the user's chat history and external documents to assist in your response.\n\n"
            "Your task is to:\n"
            "- Accurately and clearly answer the user's latest question.\n"
            "- Incorporate any relevant information from the context documents enclosed below.\n"
            # "- Reference the source(s) whenever applicable.\n"
            "- Use appropriate markdown formatting for clarity and readability (e.g., bullet points, headings, code blocks, tables).\n\n"
            "- If not available in the context, mention that and then answer from your own knowledge.\n"
            "Contextual Documents:\n"
            "<CONTEXT>{context}</CONTEXT>"
        )),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input} \n\n **Strictly stick to the instructions!**")
    ]
)


# Summarizer Template:
template_summarize = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "".join([
            "You are an expert at summarizing conversations into standalone prompts.\n"
            "You are given a complete chat history, ending with the user's latest message.\n\n"
            "Your task is to:\n"
            "- Understand the entire conversation context.\n"
            "- Identify references in the latest user message that relate to earlier messages.\n"
            "- Create a single clear, concise, and standalone question or prompt.\n"
            "- This final prompt should be fully understandable without needing the prior conversation.\n"
            "- It will be used to retrieve the most relevant documents.\n\n"
            "Only return the rewritten standalone prompt. Do not add explanations or formatting."
        ])),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}. \n\n **Make one standalone prompt as asked!**")
    ]
)


# ------------------------------------------------------------------------------
# Chains:
# ------------------------------------------------------------------------------

# 3 User Input + Chat History > Summarizer Template > Standalone Que > Get Docs
summarize_chain = create_history_aware_retriever(
    llm_retrieval, retriever, template_summarize)

# 4 Multiple Docs > Combine All > Chat Template > Final Output
qa_chain = create_stuff_documents_chain(llm, template_chat)

# 2 Input + Chat History > summarizer-chain > qa-chain > Final Output
rag_chain = create_retrieval_chain(summarize_chain, qa_chain)

# 1 Final main chain:
conversational_rag_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="messages",
    output_messages_key="answer",
)


# ------------------------------------------------------------------------------
# Actual Functions:
# ------------------------------------------------------------------------------

def get_conversational_rag_response(prompt: str, session_id: str) -> Generator[str, None, None]:
    """Returns a response in streaming mode."""

    for chunk in conversational_rag_chain.stream(
        input={"input": prompt},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk['answer'] if "answer" in chunk else ""


def embed_file(self):
    ...


if __name__ == "__main__":
    def sep():
        print(f"\n {'-' * 100} \n\n")

    # user_message = "What is Gemma-3?"
    # logger.log("Seeking response...", level='info')
    # sep()

    # # Dummy response mode:
    # print(get_response(user_message, dummy=True))
    # sep()
    # response = get_response_stream(user_message, dummy=True)
    # for chunk in response:
    #     print(chunk, end='', flush=True)
    # sep()

    # # Real Response Mode:
    # print(get_response(user_message))
    # sep()
    # response = get_response_stream(user_message)
    # for chunk in response:
    #     print(chunk, end='', flush=True)
    # sep()

    # logger.log("All tests successful", level='info')

    # Conversational RAG:
    session_id = "test-session"
    user_message = "What is Gemma-3?"
    logger.log("Seeking response...", level='info')
    sep()

    for chunk in get_conversational_rag_response(user_message, session_id):
        print(chunk, end='', flush=True)
    sep()
    for chunk in get_conversational_rag_response("Superb! What is AGI?", session_id):
        print(chunk, end='', flush=True)
    sep()
    for chunk in get_conversational_rag_response("Is it achieved yet?", session_id):
        print(chunk, end='', flush=True)
    sep()
    print(get_session_history(session_id).messages)