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
# Main:
# ------------------------------------------------------------------------------

# I don't any specific idea for what to rename this class as, for now, going with LangChain:
class LC:
    def __init__(self, model_name: str = "gemma3:latest",
                 max_context_size: int = 14000, doc_chunk_size: int = 750,
                 dummy_response: bool = False, temperature_chat: int = 1,
                 temperature_summary: int = 0.75, embedding_model: str = "mxbai-embed-large:latest"):
        """Initializes the LangChain class with the specified parameters:

        Args:
            model_name (str): Name of the model to be used.
            max_context_size (int): Maximum context size for the model (which your GPU can handle).
            doc_chunk_size (int): Size of each document chunk.
            dummy_response (bool): If True, use dummy responses instead of LLM.
            temperature_chat (int): Temperature setting for chat responses.
            temperature_summary (int): Temperature setting for summary of history for docs retrieval.
            embedding_model (str): Embedding model to be used for document retrieval.

        Returns:
            None
        """

        self.dummy = dummy_response

        self.llm = ChatOllama(
            model=model_name, num_ctx=max_context_size, temperature=temperature_chat)
        self.llm_retrieval = ChatOllama(
            model=model_name, num_ctx=max_context_size, temperature=temperature_summary)
        logger.log(
            f"LangChain initialized with model: `{model_name}`", level='info')

        self.max_context_size = max_context_size
        self.doc_chunk_size = doc_chunk_size
        self.doc_count = 3000 // doc_chunk_size

        self.chat_histories = {}
        logger.log(f"Chat histories store initialized", level='info')

        self.embedding_model = OllamaEmbeddings(model=embedding_model)
        self.database = FAISS.from_documents([Document("Hello World!")],
                                             embedding=self.embedding_model)

        self.retriever = self.database.as_retriever(
            search_type="similarity", search_kwargs={'k': 20}
        )

        logger.log(
            f"Database initialized with {embedding_model} embeddings.", level='info')

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
            logger.log(
                f"Created chat history for session id: `{session_id}`", level='info')
        return self.chat_histories[session_id]

    def init_rag():
        ...

    def get_conversational_rag_response(self, prompt: str, session_id: str) -> str:
        """Returns a response in streaming mode."""

        # Chat Template:
        template_chat = ChatPromptTemplate.from_messages(
            messages=[
                ("system",  "".join([
                    "You are a highly knowledgeable and helpful AI assistant.\n"
                    "You are provided with the user's chat history and external documents to assist in your response.\n\n"
                    "Your task is to:\n"
                    "- Accurately and clearly answer the user's latest question.\n"
                    "- Incorporate any relevant information from the context documents enclosed below.\n"
                    # "- Reference the source(s) whenever applicable.\n"
                    "- Use appropriate markdown formatting for clarity and readability (e.g., bullet points, headings, code blocks, tables).\n\n"
                    "Contextual Documents:\n"
                    "<CONTEXT>{context}</CONTEXT>"
                ])),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{input}")
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

        # For summary 15k chat + 1k system and all
        trim_summary = trim_messages(
            max_tokens=self.max_context_size - 1000,
            strategy="last", token_counter=self.llm_retrieval, start_on="human",
            allow_partial=True,  # include_system=True,
        )

        # For chat 10k chat + 5*1k docs + 1k system and all
        trim_chat = trim_messages(
            max_tokens=(self.max_context_size -
                        self.doc_chunk_size * self.doc_count - 1000),
            strategy="last", token_counter=self.llm, start_on="human",
            allow_partial=True
        )

        # Now the actual thing begins:
        # Using 3 diff chains to create the final chain:

        # 3 User Input + Chat History > Summarizer Template > Standalone Que > Get Docs
        summarize_chain = create_history_aware_retriever(
            self.llm_retrieval, self.retriever, template_summarize)

        # 4 Multiple Docs > Combine All > Chat Template > Final Output
        qa_chain = create_stuff_documents_chain(self.llm, template_chat)

        # 2 Input + Chat History > [ `Summarizer Template` > `Get Docs` ] > [ `Combine` > `Chat Template` ] > Output
        rag_chain = create_retrieval_chain(summarize_chain, qa_chain)

        # 1 Final main chain:
        conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="messages",
            output_messages_key="answer",
        )

        return conversational_rag_chain.stream(
            input={"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )

    def embed_file(self):
        ...

    # Lesson learned: Never use return and yield in the same function.
    # Python will always return generators !

    def get_response(self, prompt: str = "Hello!") -> str:
        """Returns a complete response as a string (non-streaming mode)."""

        if self.dummy:
            logger.log(f"Type: Dummy, Stream: False", level='info')
            return choice(dummy_resp)

        else:
            logger.log(f"Type: LLM, Stream: False", level='info')
            return str(self.llm.invoke(prompt).content)

    def get_response_stream(self, prompt: str = "Hello!") -> Generator[str, None, None]:
        """Yields the response in chunks (streaming mode)."""

        if self.dummy:
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
            for chunk in self.llm.stream(prompt):
                yield chunk.content


if __name__ == "__main__":
    def sep():
        print(f"\n {'-' * 100} \n\n")

    user_message = "What is Gemma-3?"
    logger.log("Seeking response...", level='info')
    sep()

    # Dummy response mode:
    lc = LC(dummy_response=True)
    # without streaming:
    print(lc.get_response(user_message))
    sep()

    # with streaming:
    response = lc.get_response_stream(user_message)
    for chunk in response:
        print(chunk, end='', flush=True)
    sep()

    # Real Response Mode:
    lc = LC(dummy_response=False)
    # without streaming:
    print(lc.get_response(user_message))
    sep()

    # with streaming:
    response = lc.get_response_stream(user_message)
    for chunk in response:
        print(chunk, end='', flush=True)
    sep()

    logger.log("All tests successful", level='info')
