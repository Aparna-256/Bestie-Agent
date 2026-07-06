from langchain_community.chat_message_histories import ChatMessageHistory

# ==========================================================
# 🎀 Bestie's Chat Memory
# ==========================================================
# Stores the conversation history during the current session.
# (This is temporary memory—it resets when you restart the app.)
# ==========================================================

chat_history = ChatMessageHistory()