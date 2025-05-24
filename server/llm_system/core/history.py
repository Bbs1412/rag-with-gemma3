from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class HistoryStore:
    def __init__(self, logger):
        self.logger = logger
        self.histories = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
            self.logger.log(
                f"Created history for session: `{session_id}`", level="info")
        return self.histories[session_id]
