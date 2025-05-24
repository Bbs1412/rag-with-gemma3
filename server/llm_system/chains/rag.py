from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory

from .prompts import template_chat as chat_prompt
from .prompts import template_summarize as summary_prompt

from typing import Callable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel


def build_rag_chain(
        llm_chat: BaseChatModel, llm_summary: BaseChatModel,
        retriever: VectorStoreRetriever, get_history_fn: Callable):
    """Builds a Conversational RAG (Retrieval-Augmented Generation) chain.

    Args:
        llm_chat (BaseChatModel): The LLM model for generating chat responses.
        llm_summary (BaseChatModel): The LLM model for summarizing chat history.
        retriever (VectorStoreRetriever): The retriever to fetch relevant documents.
        get_history_fn (Callable): Function to retrieve chat history for a session.

    Returns:
        RunnableWithMessageHistory: A runnable chain that processes user input and chat history
        to provide a final answer based on retrieved documents and chat context.
    """
    
    # Chain to summarize the history and retrieve relevant documents
    # 3 User Input + Chat History > Summarizer Template > Standalone Que > Get Docs
    retriever_chain = create_history_aware_retriever(
        llm_summary, retriever, summary_prompt)

    # Chain to combine the retrieved documents and get the final answer
    # 4 Multiple Docs > Combine All > Chat Template > Final Output
    qa_chain = create_stuff_documents_chain(llm=llm_chat, prompt=chat_prompt)

    # Main RAG Chain:
    # 2 Input + Chat History > [ `Summarizer Template` > `Get Docs` ] > [ `Combine` > `Chat Template` ] > Output
    rag_chain = create_retrieval_chain(retriever_chain, qa_chain)

    # 1 Final Conversational RAG Chain:
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
