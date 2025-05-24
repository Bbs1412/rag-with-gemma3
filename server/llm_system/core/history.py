from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class HistoryStore:
    """A class to manage chat message histories for different sessions/users.

    Args:
        logger (logging.Logger): Logger instance for logging history operations.

    ## Functions:
        + `get_session_history(session_id: str)`: Retrieves the chat history for a given session ID.
            If no history exists, it creates a new one and returns empty initialized history.
    """

    def __init__(self, logger):
        self.logger = logger
        self.histories = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
            self.logger.log(
                f"Created history for session: `{session_id}`", level="info")

        self.logger.log(
            f"Retrieved history for session: `{session_id}`", level="info")
        return self.histories[session_id]
