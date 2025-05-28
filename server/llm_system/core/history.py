from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from logger import get_logger
log = get_logger(name="core_history")


class HistoryStore:
    """A class to manage chat message histories for different sessions/users.

    ## Functions:
        + `get_session_history(session_id: str)`: Retrieves the chat history for a given session ID.
            If no history exists, it creates a new one and returns empty initialized history.
    """

    def __init__(self):
        self.histories = {}
        log.info("Initialized HistoryStore with an empty history dictionary.")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
            log.info(f"Created history for session: `{session_id}`")

        log.info(f"Retrieved history for session: `{session_id}`")
        return self.histories[session_id]
