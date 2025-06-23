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
        """Retrieves the chat history for a given session ID.
        If no history exists for the session, it creates a new one and returns it.

        Args:
            session_id (str): The unique identifier for the chat session.
        Returns:
            BaseChatMessageHistory: The chat message history for the session.
        """

        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
            log.info(f"Created history for session: `{session_id}`")

        log.info(f"Retrieved history for session: `{session_id}`")
        return self.histories[session_id]

    def clear_session_history(self, session_id: str):
        """Clears the chat history for a given session ID.

        Args:
            session_id (str): The unique identifier for the chat session.
        Returns:
            bool: True if the history was cleared, False if no history existed for the session.
        """

        if session_id in self.histories:
            del self.histories[session_id]
            log.info(f"Cleared history for session: `{session_id}`")
            return True
        else:
            log.warning(f"No history found for session: `{session_id}` to clear.")
            return False
