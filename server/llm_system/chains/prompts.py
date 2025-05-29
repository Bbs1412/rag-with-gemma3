from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from logger import get_logger
log = get_logger(name="chains_prompts")

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
        MessagesPlaceholder(variable_name="chat_history"),
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
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}. \n\n **Make one standalone prompt as asked!**")
    ]
)

log.info("Initialized chat and summarize prompt templates.")